import argparse
import base64
import html
import re
from email.utils import parseaddr
from pathlib import Path

from authentication import authenticate_gmail


DEFAULT_DOMAIN = "pusan.ac.kr"
DEFAULT_OUTPUT_DIR = Path("emails")
DEFAULT_AFTER_DATE = "2025/03/01"


def decode_body(data):
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    raw = base64.urlsafe_b64decode(data + padding)
    return raw.decode("utf-8", errors="replace")


def html_to_text(value):
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p>", "\n\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def iter_parts(payload):
    yield payload
    for part in payload.get("parts", []) or []:
        yield from iter_parts(part)


def extract_text(payload):
    plain_parts = []
    html_parts = []

    for part in iter_parts(payload):
        filename = part.get("filename")
        body = part.get("body", {})
        data = body.get("data")
        if filename or not data:
            continue

        mime_type = part.get("mimeType", "")
        text = decode_body(data).strip()
        if not text:
            continue

        if mime_type == "text/plain":
            plain_parts.append(text)
        elif mime_type == "text/html":
            html_parts.append(html_to_text(text))

    if plain_parts:
        return "\n\n".join(plain_parts).strip()
    if html_parts:
        return "\n\n".join(part for part in html_parts if part).strip()
    return ""


def header_map(payload):
    return {
        header.get("name", "").lower(): header.get("value", "")
        for header in payload.get("headers", [])
    }


def safe_filename(value, fallback):
    value = value.strip() or fallback
    value = re.sub(r"[\\/:*?\"<>|]", "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value[:120].strip(" ._")
    return value or fallback


def list_message_ids(service, query, include_spam_trash):
    message_ids = []
    request = service.users().messages().list(
        userId="me",
        q=query,
        includeSpamTrash=include_spam_trash,
        maxResults=500,
    )

    while request is not None:
        response = request.execute()
        message_ids.extend(item["id"] for item in response.get("messages", []))
        request = service.users().messages().list_next(request, response)

    return message_ids


def existing_message_ids(output_dir):
    ids = set()
    for path in output_dir.glob("*.txt"):
        match = re.search(r"_([0-9a-f]{16,})\.txt$", path.name)
        if match:
            ids.add(match.group(1))
    return ids


def download_emails(domain, output_dir, include_spam_trash, after_date):
    service = authenticate_gmail()
    query = f"from:{domain} -in:sent after:{after_date}"
    print(f"searching Gmail with query: {query}", flush=True)
    message_ids = list_message_ids(service, query, include_spam_trash)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = existing_message_ids(output_dir)

    saved = 0
    skipped = 0
    skipped_sent = 0
    already_saved = 0
    start_number = len(existing_ids)

    for index, message_id in enumerate(message_ids, start=1):
        if message_id in existing_ids:
            already_saved += 1
            continue

        message = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full",
        ).execute()
        if "SENT" in message.get("labelIds", []):
            skipped_sent += 1
            continue

        payload = message.get("payload", {})
        headers = header_map(payload)
        from_header = headers.get("from", "")
        from_address = parseaddr(from_header)[1].lower()

        if not from_address.endswith(domain.lower()):
            skipped += 1
            continue

        subject = headers.get("subject", "(no subject)")
        body = extract_text(payload)
        file_stem = safe_filename(subject, message_id)
        file_path = output_dir / f"{start_number + saved + 1:04d}_{file_stem}_{message_id}.txt"

        text = "\n".join([
            f"Subject: {subject}",
            f"From: {from_header}",
            f"To: {headers.get('to', '')}",
            f"Cc: {headers.get('cc', '')}",
            f"Date: {headers.get('date', '')}",
            f"Gmail Message ID: {message_id}",
            "",
            body,
            "",
        ])
        file_path.write_text(text, encoding="utf-8")
        saved += 1

        if saved % 25 == 0:
            print(f"saved {saved} new emails...", flush=True)

    return len(message_ids), saved, skipped, skipped_sent, already_saved


def main():
    parser = argparse.ArgumentParser(
        description="Download Gmail messages from senders ending with a domain."
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--after",
        default=DEFAULT_AFTER_DATE,
        help="Gmail date filter in YYYY/MM/DD format. Default: 2025/03/01.",
    )
    parser.add_argument(
        "--exclude-spam-trash",
        action="store_true",
        help="Do not search Spam and Trash.",
    )
    args = parser.parse_args()

    total, saved, skipped, skipped_sent, already_saved = download_emails(
        domain=args.domain,
        output_dir=Path(args.output_dir),
        include_spam_trash=not args.exclude_spam_trash,
        after_date=args.after,
    )
    print(f"query candidates: {total}")
    print(f"already saved: {already_saved}")
    print(f"saved new: {saved}")
    print(f"skipped sent messages: {skipped_sent}")
    print(f"skipped after From-domain check: {skipped}")


if __name__ == "__main__":
    main()
