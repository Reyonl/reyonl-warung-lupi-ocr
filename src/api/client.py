"""
Laravel API Client for Warung Lupi backend.

Provides access to existing endpoints:
- GET /api/products — list products for matching (Phase 3)
- GET /api/customers — list customers for matching (Phase 4)
- POST /api/transactions — create new transaction (Phase 9: finalize draft)

This does NOT modify or add endpoints to the Laravel backend.
Uses the existing API contract exactly as-is.

Authentication: Bearer token (Laravel Sanctum)
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class WarungLupiAPIError(Exception):
    """Exception raised when API calls fail."""
    pass


class WarungLupiAPIClient:
    """
    Client for Warung Lupi Laravel backend API.

    Usage:
        client = WarungLupiAPIClient(base_url="http://localhost:8000", token="...")
        products = client.get_products()
        customers = client.get_customers()
        transaction_id = client.create_transaction(draft_payload)
    """

    def __init__(self, base_url: str, api_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.api_token = api_token

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        query_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API."""
        url = urljoin(self.base_url, endpoint.lstrip("/"))

        if query_params:
            from urllib.parse import urlencode
            url += "?" + urlencode(query_params)

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
                return json.loads(content)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"API error {e.code}: {error_body}")
            raise WarungLupiAPIError(
                f"API request failed: {e.code} - {error_body[:500]}"
            )
        except urllib.error.URLError as e:
            logger.error(f"Connection error: {e}")
            raise WarungLupiAPIError(f"Cannot connect to Warung Lupi API: {e}")

    def check_health(self) -> bool:
        """Check if API is reachable."""
        try:
            self._make_request("api/health")
            return True
        except WarungLupiAPIError:
            try:
                # Fallback: try Sanctum token check
                self._make_request("api/user")
                return True
            except WarungLupiAPIError:
                return False

    def get_products(self) -> List[Dict[str, Any]]:
        """
        Fetch all products from Warung Lupi database (Phase 3 context).

        Uses existing endpoint: GET /api/products
        Returns list of product dicts with at least: id, name, default_price, unit.
        """
        try:
            resp = self._make_request("api/products")
            # Handle different response shapes
            if isinstance(resp, list):
                return resp
            elif isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            elif isinstance(resp, dict) and "products" in resp:
                return resp["products"]
            else:
                logger.warning(f"Unexpected products response shape: {type(resp)}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch products: {e}")
            return []

    def get_customers(self) -> List[Dict[str, Any]]:
        """
        Fetch all customers from Warung Lupi database (Phase 4 context).

        Uses existing endpoint: GET /api/customers
        Returns list of customer dicts with at least: id, name.
        """
        try:
            resp = self._make_request("api/customers")
            if isinstance(resp, list):
                return resp
            elif isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            elif isinstance(resp, dict) and "customers" in resp:
                return resp["customers"]
            else:
                logger.warning(f"Unexpected customers response shape: {type(resp)}")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch customers: {e}")
            return []

    def search_customers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search customers by name (Phase 4 — supplementary to matching).

        Uses existing endpoint: GET /api/customers?search={query}
        """
        try:
            resp = self._make_request(
                "api/customers",
                query_params={"search": query}
            )
            if isinstance(resp, list):
                return resp
            elif isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            return []
        except Exception as e:
            logger.warning(f"Customer search failed: {e}")
            return []

    def search_products(self, query: str) -> List[Dict[str, Any]]:
        """
        Search products by name (Phase 3 — supplementary to matching).

        Uses existing endpoint: GET /api/products?search={query}
        """
        try:
            resp = self._make_request(
                "api/products",
                query_params={"search": query}
            )
            if isinstance(resp, list):
                return resp
            elif isinstance(resp, dict) and "data" in resp:
                return resp["data"]
            return []
        except Exception as e:
            logger.warning(f"Product search failed: {e}")
            return []

    def create_transaction(
        self,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new transaction in Warung Lupi (Phase 9: finalize draft).

        Uses existing endpoint: POST /api/transactions
        Payload format (from DraftTransactionService.confirm_draft()):
            {
                "date": "2026-10-08",
                "customer_id": 14,
                "customer_name": "Rama",
                "items": [
                    {"product_id": 1, "unit_price": 5000, "quantity": 2, "subtotal": 10000}
                ],
                "total_amount": 10000,
                "notes": "...",
            }

        API expects (Laravel backend contract):
            {
                "customer_id": int,
                "date": "Y-m-d",
                "items": [{"product_id": int, "quantity": int, "unit_price": float}]
            }

        Returns the API response (transaction ID + created data).
        """
        # Map our draft payload to the Laravel API format
        api_payload = {
            "date": payload.get("date"),
            "customer_id": payload.get("customer_id"),
            "customer_name": payload.get("customer_name"),
            "items": [],
        }

        for item in payload.get("items", []):
            api_payload["items"].append({
                "product_id": item["product_id"],
                "unit_price": item["unit_price"],
                "quantity": item["quantity"],
                "subtotal": item.get("subtotal", 0),
                "notes": item.get("notes", ""),
            })

        if payload.get("notes"):
            api_payload["notes"] = payload["notes"]
        if payload.get("total_amount"):
            api_payload["total_amount"] = payload["total_amount"]

        return self._make_request("api/transactions", method="POST", data=api_payload)

    def add_transaction_items(
        self,
        transaction_id: int,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add items to a transaction (if using the multi-step flow).

        Uses existing endpoint: POST /api/transactions/{id}/items
        """
        api_items = []
        for item in items:
            api_items.append({
                "product_id": item["product_id"],
                "unit_price": item.get("unit_price", 0),
                "quantity": item.get("quantity", 1),
                "subtotal": item.get("subtotal", item.get("unit_price", 0) * item.get("quantity", 1)),
            })

        return self._make_request(
            f"api/transactions/{transaction_id}/items",
            method="POST",
            data={"items": api_items},
        )

    def update_transaction(
        self,
        transaction_id: int,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update a transaction.

        Uses existing endpoint: PUT /api/transactions/{id}
        """
        return self._make_request(
            f"api/transactions/{transaction_id}",
            method="PUT",
            data=data,
        )


def load_api_config(config_path: str) -> WarungLupiAPIClient:
    """
    Load API configuration from a JSON config file.

    Config format:
        {
            "base_url": "http://localhost:8000",
            "api_token": "1|abc123..."
        }
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return WarungLupiAPIClient(
        base_url=config["base_url"],
        api_token=config.get("api_token"),
    )
