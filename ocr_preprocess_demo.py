from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IMAGE_PATH = "data/course_screenshot.png"
DEFAULT_OUTPUT_PATH = "data/course_screenshot_ocr.txt"
DEFAULT_GOOGLE_VISION_KEY_PATH = Path(__file__).resolve().parent / "private" / "google_vision_key.json"


def _json_safe(value: Any) -> Any:
    """Convert Paddle / numpy objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def preprocess_image(image_path: str, scale: int = 2):
    """Open and lightly preprocess a screenshot for Tesseract/EasyOCR.

    PaddleOCR usually works better on the original screenshot, so PaddleOCR does
    not use this function by default.
    """
    from PIL import Image, ImageOps  # type: ignore

    image = Image.open(image_path)
    if scale and scale != 1:
        image = image.resize((image.width * scale, image.height * scale))
    grayscale = ImageOps.grayscale(image)
    return ImageOps.autocontrast(grayscale)


def _mapping_from_paddle_page(page: Any) -> dict[str, Any]:
    """Normalize PaddleOCR 3.x result page into a dict when possible."""
    if isinstance(page, dict):
        return page

    # PaddleOCR result objects often support item access even when they are not
    # plain dicts.
    result: dict[str, Any] = {}
    for key in (
        "rec_texts",
        "rec_scores",
        "rec_boxes",
        "dt_polys",
        "rec_polys",
        "text",
        "ocr_res",
    ):
        try:
            value = page[key]
            result[key] = value
        except Exception:
            pass

    if result:
        return result

    for method_name in ("to_dict", "json", "dict"):
        method = getattr(page, method_name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    try:
                        loaded = json.loads(value)
                        if isinstance(loaded, dict):
                            return loaded
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

    if hasattr(page, "__dict__"):
        data = {k: v for k, v in vars(page).items() if not k.startswith("_")}
        if data:
            return data

    return {"raw": str(page)}


def _extract_texts_from_old_paddle_result(result: Any) -> list[str]:
    """Support older PaddleOCR 2.x style output.

    Old style commonly looks like:
    [ [ [box, (text, score)], [box, (text, score)] ] ]
    """
    texts: list[str] = []
    if not isinstance(result, list):
        return texts

    def walk(value: Any) -> None:
        if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
            texts.append(value[0])
            return
        if isinstance(value, list):
            # One OCR item: [box, (text, score)]
            if len(value) >= 2 and isinstance(value[1], (tuple, list)) and value[1]:
                if isinstance(value[1][0], str):
                    texts.append(value[1][0])
                    return
            for item in value:
                walk(item)

    walk(result)
    return [text.strip() for text in texts if str(text).strip()]


def _extract_paddle_text_and_raw(result: Any) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    raw_pages: list[dict[str, Any]] = []

    if not isinstance(result, list):
        result = [result]

    for page in result:
        page_dict = _mapping_from_paddle_page(page)
        rec_texts = page_dict.get("rec_texts") or []

        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]
        if rec_texts:
            texts.extend(str(item).strip() for item in rec_texts if str(item).strip())
        else:
            # Fallback for old style nested result, or unusual page object.
            old_texts = _extract_texts_from_old_paddle_result(page)
            if old_texts:
                texts.extend(old_texts)
            elif page_dict.get("text"):
                texts.append(str(page_dict["text"]).strip())

        raw_pages.append(_json_safe(page_dict))

    return texts, raw_pages


def _run_paddleocr(image_path: str) -> tuple[str, dict[str, Any]]:
    """Run PaddleOCR 3.x or fallback-compatible PaddleOCR.

    Supports the newer API:
        PaddleOCR(lang="ch", use_textline_orientation=True).predict(image_path)

    It also falls back to the older API if needed, but never passes cls=True to
    predict(), because PaddleOCR 3.x does not accept it.
    """
    # These flags must be set before Paddle imports its inference backend. If
    # Paddle was already imported, they may not take effect, but they do not hurt.
    os.environ.setdefault("FLAGS_use_mkldnn", "False")
    os.environ.setdefault("FLAGS_use_onednn", "False")

    from paddleocr import PaddleOCR  # type: ignore

    start = time.perf_counter()

    # PaddleOCR versions differ in accepted constructor arguments. Try the new
    # names first, then gracefully fall back.
    try:
        ocr = PaddleOCR(lang="ch", use_textline_orientation=True)
    except TypeError:
        try:
            ocr = PaddleOCR(lang="ch", use_angle_cls=True)
        except TypeError:
            ocr = PaddleOCR(lang="ch")

    try:
        result = ocr.predict(
            image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    except TypeError:
        try:
            result = ocr.predict(image_path)
        except AttributeError:
            # PaddleOCR 2.x old API.
            result = ocr.ocr(image_path, cls=True)
    except AttributeError:
        result = ocr.ocr(image_path, cls=True)

    texts, raw_pages = _extract_paddle_text_and_raw(result)

    raw_path = Path(image_path).with_name(f"{Path(image_path).stem}_ocr_raw.json")
    raw_path.write_text(json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")

    text = "\n".join(texts)
    return text, {
        "backend": "paddleocr",
        "image_size": None,
        "ocr_seconds": round(time.perf_counter() - start, 3),
        "raw_json_path": str(raw_path),
    }


def _run_tesseract(image_path: str) -> tuple[str, dict[str, Any]]:
    import pytesseract  # type: ignore

    image = preprocess_image(image_path)
    start = time.perf_counter()
    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6",
        timeout=30,
    )
    return text, {
        "backend": "tesseract",
        "image_size": image.size,
        "ocr_seconds": round(time.perf_counter() - start, 3),
    }


def _easyocr_reconstruct_lines(results: list[Any]) -> str:
    rows: list[dict[str, Any]] = []
    for item in results:
        if len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        if not text:
            continue
        try:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
        except Exception:
            continue
        rows.append(
            {
                "text": text,
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "height": max(ys) - min(ys),
            }
        )
    if not rows:
        return ""

    rows.sort(key=lambda row: (row["y"], row["x"]))
    median_height = sorted(row["height"] for row in rows)[len(rows) // 2] or 20
    y_threshold = max(12, median_height * 0.65)

    lines: list[list[dict[str, Any]]] = []
    for row in rows:
        if not lines or abs(lines[-1][0]["y"] - row["y"]) > y_threshold:
            lines.append([row])
        else:
            lines[-1].append(row)

    reconstructed: list[str] = []
    for line in lines:
        line.sort(key=lambda row: row["x"])
        reconstructed.append(" ".join(row["text"] for row in line))
    return "\n".join(reconstructed)


def _run_easyocr(image_path: str) -> tuple[str, dict[str, Any]]:
    import easyocr  # type: ignore
    import numpy as np  # type: ignore

    image = preprocess_image(image_path)
    start = time.perf_counter()
    try:
        reader = easyocr.Reader(["en", "ch_tra"], gpu=False)
    except Exception:
        reader = easyocr.Reader(["en"], gpu=False)
    pieces = reader.readtext(np.array(image), detail=1, paragraph=False)
    text = _easyocr_reconstruct_lines(pieces)
    return text, {
        "backend": "easyocr",
        "image_size": image.size,
        "ocr_seconds": round(time.perf_counter() - start, 3),
    }


def _google_vision_key_path() -> str:
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_VISION_KEY_PATH")
    if env_path and Path(env_path).expanduser().exists():
        return str(Path(env_path).expanduser())
    if DEFAULT_GOOGLE_VISION_KEY_PATH.exists():
        return str(DEFAULT_GOOGLE_VISION_KEY_PATH)
    return ""


def _run_google_vision(image_path: str) -> tuple[str, dict[str, Any]]:
    from google.cloud import vision  # type: ignore
    from google.oauth2 import service_account  # type: ignore

    start = time.perf_counter()
    key_path = _google_vision_key_path()
    if key_path:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        credential_source = str(Path(key_path).relative_to(Path(__file__).resolve().parent)) if Path(key_path).is_relative_to(Path(__file__).resolve().parent) else key_path
    else:
        client = vision.ImageAnnotatorClient()
        credential_source = "GOOGLE_APPLICATION_CREDENTIALS/default"

    content = Path(image_path).read_bytes()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    text = ""
    if response.full_text_annotation and response.full_text_annotation.text:
        text = response.full_text_annotation.text
    elif response.text_annotations:
        text = response.text_annotations[0].description or ""

    return text, {
        "backend": "google_vision",
        "image_size": None,
        "ocr_seconds": round(time.perf_counter() - start, 3),
        "credential_source": credential_source,
    }


def _course_code_like_count(text: str) -> int:
    pattern = re.compile(r"\b[A-Z]{2,6}\s*\d{4}\b", flags=re.IGNORECASE)
    return len(pattern.findall(text or ""))


def run_ocr(
    image_path: str = DEFAULT_IMAGE_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    backend_order: Iterable[str] = ("google_vision", "paddleocr", "tesseract", "easyocr"),
) -> dict[str, Any]:
    """Run screenshot OCR in the Python (OCR) environment and save a text cache.

    The main HW2 notebook can then read `output_path` without importing OCR
    packages. This supports Google Vision, PaddleOCR, Tesseract, and EasyOCR when available.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "image_path": image_path,
        "output_path": str(output),
        "tesseract_path": shutil.which("tesseract"),
        "google_vision_key_path": _google_vision_key_path(),
        "backend": "",
        "image_size": None,
        "ocr_seconds": 0.0,
        "text_length": 0,
        "course_code_like_count": 0,
        "success": False,
        "error": "",
    }

    print(f'tesseract path: {result["tesseract_path"]}')

    runners = {
        "google": _run_google_vision,
        "google_vision": _run_google_vision,
        "cloud_vision": _run_google_vision,
        "paddleocr": _run_paddleocr,
        "tesseract": _run_tesseract,
        "easyocr": _run_easyocr,
    }

    text = ""
    errors: list[str] = []
    for backend_name in backend_order:
        runner = runners.get(str(backend_name).lower())
        if runner is None:
            continue
        try:
            print(f"trying OCR backend: {backend_name}")
            text, backend_result = runner(image_path)
            result.update(backend_result)
            result["text_length"] = len(text)
            result["course_code_like_count"] = _course_code_like_count(text)
            result["success"] = True
            print(f"OCR backend used: {backend_name}")
            break
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
            print(f"{backend_name} OCR error: {exc}")

    if not result["success"]:
        text = (
            "OCR failed. Please check OCR dependencies in the Python (OCR) kernel. "
            "PaddleOCR, Tesseract, or EasyOCR can be used. Errors: "
            + " | ".join(errors)
        )
        result["error"] = " | ".join(errors)

    output.write_text(text, encoding="utf-8")
    print(f'image size: {result["image_size"]}')
    print(f'OCR seconds: {result["ocr_seconds"]}')
    print(f'text length: {len(text)}')
    print(f'course-code-like count: {result["course_code_like_count"]}')
    if result["success"] and result["course_code_like_count"] == 0:
        print("warning: OCR succeeded but no course-code-like text was found. For demo, use a cropped normalized_course_code column screenshot.")
    print(f'first 1000 chars: {repr(text[:1000])}')
    print(f'output path: {output.resolve()}')
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR for NTHU COPILOT screenshot demo.")
    parser.add_argument("image_path", nargs="?", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--backend-order",
        default="paddleocr,tesseract,easyocr",
        help="Comma-separated backend priority, e.g. paddleocr,easyocr,tesseract",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    order = [item.strip() for item in args.backend_order.split(",") if item.strip()]
    run_ocr(args.image_path, output_path=args.output, backend_order=order)
