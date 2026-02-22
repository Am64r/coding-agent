from pathlib import Path
from ..task import EvalTask
from ..verifier import TestsPasses

_HIDDEN_TESTS = """\
import json
import requests as _requests
from unittest.mock import patch, MagicMock
from api_client import APIClient, APIError

def _mock_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp

def _find_session(client):
    for v in vars(client).values():
        if isinstance(v, _requests.Session):
            return v
    return None

def test_get_request():
    client = APIClient("https://api.example.com", token="abc123")
    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(200, {"id": 1})
        result = client.get("/users/1")
        assert result == {"id": 1}
        call_kwargs = mock_req.call_args[1]
        per_req = call_kwargs.get("headers", {}).get("Authorization")
        session = _find_session(client)
        session_level = session.headers.get("Authorization") if session else None
        assert (per_req or session_level) == "Bearer abc123"

def test_post_request():
    client = APIClient("https://api.example.com", token="xyz")
    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(201, {"id": 2, "name": "Alice"})
        result = client.post("/users", json={"name": "Alice"})
        assert result["name"] == "Alice"

def test_api_error_on_404():
    client = APIClient("https://api.example.com")
    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(404, {"error": "not found"})
        try:
            client.get("/missing")
            assert False, "Should have raised APIError"
        except APIError as e:
            assert e.status_code == 404

def test_retry_on_500():
    client = APIClient("https://api.example.com", retries=3)
    with patch("requests.Session.request") as mock_req:
        mock_req.side_effect = [
            _mock_response(500),
            _mock_response(500),
            _mock_response(200, {"ok": True}),
        ]
        result = client.get("/flaky")
        assert result == {"ok": True}
        assert mock_req.call_count == 3

def test_retry_exhausted():
    client = APIClient("https://api.example.com", retries=2)
    with patch("requests.Session.request") as mock_req:
        mock_req.side_effect = [
            _mock_response(500),
            _mock_response(500),
        ]
        try:
            client.get("/always-fail")
            assert False, "Should have raised APIError"
        except APIError as e:
            assert e.status_code == 500

def test_put_and_delete():
    client = APIClient("https://api.example.com", token="t")
    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(200, {"updated": True})
        assert client.put("/item/1", json={"v": 2}) == {"updated": True}

    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(204, {})
        result = client.delete("/item/1")
        assert result == {}

def test_base_url_joining():
    client = APIClient("https://api.example.com/v1")
    with patch("requests.Session.request") as mock_req:
        mock_req.return_value = _mock_response(200, {})
        client.get("/users")
        call_args = mock_req.call_args
        url = call_args[1].get("url") or (call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "api.example.com" in url
"""

def setup(workspace: Path) -> None:
    (workspace / "test_api_client.py").write_text(_HIDDEN_TESTS)

task = EvalTask(
    id="rest_api_client",
    prompt=(
        "Create api_client.py with:\n\n"
        "1. APIError(Exception) — custom exception with a `status_code` attribute and a `response` attribute.\n\n"
        "2. APIClient(base_url, token=None, retries=1) — a REST client using the `requests` library:\n"
        "   - Uses a requests.Session internally\n"
        "   - If token is provided, adds 'Authorization: Bearer <token>' header to all requests\n"
        "   - Methods: get(path, **kwargs), post(path, **kwargs), put(path, **kwargs), delete(path, **kwargs)\n"
        "   - Each method makes the appropriate HTTP request to base_url + path\n"
        "   - On 5xx responses, retry up to `retries` times (total attempts = retries)\n"
        "   - On 4xx responses, raise APIError immediately with the status code\n"
        "   - On success, return response.json()\n\n"
        "Use the `requests` library (already installed)."
    ),
    setup=setup,
    verify=TestsPasses("python3 -m pytest test_api_client.py -v").check,
    tags=["api", "design-pattern", "python", "hidden-tests"],
)
