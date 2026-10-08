"""
Phase 5: Draft Transaction Service.

Creates a draft transaction from Phase 2 interpretation + Phase 3/4 matching.
The draft is NOT a finalized transaction — it must be confirmed by user.

Key responsibilities:
- Convert InterpretedResult into DraftTransaction
- Apply matched products/customers (from Phase 3/4)
- Flag items that need user review
- Preserve undo history for correction flow
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from src.interpreter.service import InterpretedResult, InterpretedItem

logger = logging.getLogger(__name__)

@dataclass
class DraftTransactionItem:
    """A single item in a draft transaction."""
    id: str  # UUID for client-side editing
    description: str  # Display name (detected or matched product name)
    product_id: Optional[int] = None  # ID in Warung Lupi product DB
    quantity: int = 1
    unit_price: float = 0.0
    unit: str = "pcs"
    subtotal: float = 0.0  # quantity * unit_price
    confidence: float = 0.0  # how confident we are in this item
    is_manual: bool = False  # if True, user added/edited manually
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "unit": self.unit,
            "subtotal": self.subtotal,
            "confidence": round(self.confidence, 4),
            "is_manual": self.is_manual,
            "notes": self.notes,
            "warnings": self.warnings,
        }


@dataclass
class DraftTransaction:
    """Draft transaction — NOT a finalized transaction."""
    id: str  # UUID for this draft
    date: Optional[str] = None  # ISO format: "2026-10-08"
    customer_id: Optional[int] = None
    customer_name: str = ""
    items: List[DraftTransactionItem] = field(default_factory=list)
    notes: str = ""
    status: str = "draft"  # draft | confirmed | cancelled
    created_at: str = ""
    interpretation_log: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_quantity(self) -> int:
        return sum(item.quantity for item in self.items)

    @property
    def total_amount(self) -> float:
        return sum(item.subtotal for item in self.items)

    @property
    def has_unresolved_warnings(self) -> bool:
        """Check if any item has warnings that need user attention."""
        return any(
            len(item.warnings) > 0 or item.confidence < 0.5
            for item in self.items
        )

    @property
    def unresolved_items(self) -> List[DraftTransactionItem]:
        """Items that need user review."""
        return [
            item for item in self.items
            if item.confidence < 0.5 or len(item.warnings) > 0
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "date": self.date,
            "customer_id": self.customer_id,
            "customer_name": self.customer_name,
            "items": [item.to_dict() for item in self.items],
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at,
            "total_quantity": self.total_quantity,
            "total_amount": self.total_amount,
            "has_unresolved_warnings": self.has_unresolved_warnings,
            "unresolved_count": len(self.unresolved_items),
            "interpretation_log": self.interpretation_log,
        }


class DraftTransactionService:
    """Creates and manages draft transactions from AI interpretation."""

    def __init__(self):
        pass

    def create_draft_from_interpretation(
        self,
        interpretation: InterpretedResult,
    ) -> DraftTransaction:
        """Convert Phase 2 interpretation into a Phase 5 draft transaction."""
        draft = DraftTransaction(
            id=str(uuid4()),
            date=interpretation.date,
            created_at=datetime.now().isoformat(),
        )

        # Set customer
        if interpretation.customer:
            draft.customer_name = interpretation.customer.detected_name or ""
            if interpretation.customer.matched_customer_id:
                draft.customer_id = interpretation.customer.matched_customer_id
            elif interpretation.customer.customer_candidates:
                top_candidate = interpretation.customer.customer_candidates[0]
                if top_candidate["score"] < 0.7:
                    draft.customer_name = f"{interpretation.customer.detected_name} (belum dipastikan)"

        # Build items
        for item in interpretation.items:
            draft_item = self._create_draft_item(item)
            draft.items.append(draft_item)

        # Set notes from interpretation
        if interpretation.notes:
            draft.notes = interpretation.notes

        # Preserve audit info (Phase 7)
        draft.interpretation_log = {
            "source_scan": interpretation.source_scan,
            "date_confidence": interpretation.date_confidence,
            "customer": {
                "detected_name": interpretation.customer.detected_name if interpretation.customer else None,
                "confidence": interpretation.customer.confidence if interpretation.customer else 0,
                "matched_id": interpretation.customer.matched_customer_id if interpretation.customer else None,
                "candidates": interpretation.customer.customer_candidates if interpretation.customer else [],
            },
            "total_confidence": interpretation.total_confidence,
            "uncertain": interpretation.uncertain,
            "items_before_matching": len(interpretation.items),
        }

        return draft

    def _create_draft_item(self, item: InterpretedItem) -> DraftTransactionItem:
        """Create a draft item from an interpreted item."""
        draft_item = DraftTransactionItem(
            id=str(uuid4()),
            description=item.detected_text,
            quantity=item.quantity if item.quantity is not None else 1,
            unit_price=float(item.price) if item.price is not None else 0.0,
            unit="pcs",
            confidence=item.confidence,
        )

        if item.matched_product_id:
            draft_item.product_id = item.matched_product_id
            draft_item.description = item.matched_product_name

        draft_item.subtotal = draft_item.quantity * draft_item.unit_price

        if not item.matched_product_id:
            draft_item.warnings.append("produk_tidak_ditemukan — pilih manual")
        if item.quantity is None:
            draft_item.warnings.append("quantity_tidak_terdeteksi")
        if item.price is None:
            draft_item.warnings.append("harga_tidak_terdeteksi")
        if item.struck_through:
            draft_item.warnings.append("coretan — periksa ulang")
        for uf in item.uncertain_fields:
            draft_item.warnings.append(f"uncertain_{uf}")

        return draft_item

    def apply_product_defaults(
        self, draft: DraftTransaction, products: List[dict],
    ) -> DraftTransaction:
        """Apply default prices and units from product database."""
        product_lookup = {p["id"]: p for p in products}
        for item in draft.items:
            if item.product_id and item.product_id in product_lookup:
                prod = product_lookup[item.product_id]
                if item.unit_price == 0.0:
                    item.unit_price = float(prod.get("default_price", 0.0))
                if prod.get("unit"):
                    item.unit = prod["unit"]
                item.subtotal = item.quantity * item.unit_price
        return draft

    def update_item_quantity(
        self, draft: DraftTransaction, item_id: str, quantity: int,
    ) -> DraftTransaction:
        """Update an item's quantity (Phase 6: user editing)."""
        for item in draft.items:
            if item.id == item_id:
                item.quantity = max(1, quantity)
                item.is_manual = True
                item.subtotal = item.quantity * item.unit_price
                break
        return draft

    def update_item_price(
        self, draft: DraftTransaction, item_id: str, price: float,
    ) -> DraftTransaction:
        """Update an item's price (Phase 6: user editing)."""
        for item in draft.items:
            if item.id == item_id:
                item.unit_price = price
                item.is_manual = True
                item.subtotal = item.quantity * price
                break
        return draft

    def update_item_product(
        self, draft: DraftTransaction, item_id: str,
        product_id: int, product_name: str,
    ) -> DraftTransaction:
        """Update an item's product (Phase 6: user selects product)."""
        for item in draft.items:
            if item.id == item_id:
                item.product_id = product_id
                item.description = product_name
                item.is_manual = True
                item.warnings = [
                    w for w in item.warnings
                    if "produk_tidak_ditemukan" not in w
                ]
                break
        return draft

    def add_manual_item(
        self, draft: DraftTransaction, description: str,
        quantity: int = 1, unit_price: float = 0.0,
        product_id: Optional[int] = None, unit: str = "pcs",
    ) -> DraftTransaction:
        """Add a new item manually (Phase 6: user adding item)."""
        draft_item = DraftTransactionItem(
            id=str(uuid4()),
            description=description,
            product_id=product_id,
            quantity=quantity,
            unit_price=unit_price,
            unit=unit,
            subtotal=quantity * unit_price,
            confidence=1.0,
            is_manual=True,
        )
        draft.items.append(draft_item)
        return draft

    def remove_item(
        self, draft: DraftTransaction, item_id: str,
    ) -> DraftTransaction:
        """Remove an item (Phase 6: user deletes item)."""
        draft.items = [item for item in draft.items if item.id != item_id]
        return draft

    def change_customer(
        self, draft: DraftTransaction, customer_id: int, customer_name: str,
    ) -> DraftTransaction:
        """Change customer (Phase 6: user selects customer)."""
        draft.customer_id = customer_id
        draft.customer_name = customer_name
        return draft

    def confirm_draft(self, draft: DraftTransaction) -> Dict[str, Any]:
        """Phase 6 -> Phase 9: Convert draft to Laravel API payload."""
        draft.status = "confirmed"

        transaction_payload = {
            "customer_id": draft.customer_id,
            "transaction_date": draft.date,
            "notes": draft.notes,
        }

        items_payload = []
        for item in draft.items:
            unit_price_int = int(round(item.unit_price))
            qty = max(1, item.quantity)
            items_payload.append({
                "product_id": item.product_id,
                "product_name": item.description,
                "description": item.notes if item.notes else None,
                "quantity": qty,
                "unit": item.unit,
                "unit_price": unit_price_int,
                "subtotal": int(round(qty * unit_price_int)),
            })

        return {
            "transaction": transaction_payload,
            "items": items_payload,
            "total_amount": int(round(draft.total_amount)),
            "customer_name": draft.customer_name,
            "draft_id": draft.id,
            "has_unresolved_warnings": draft.has_unresolved_warnings,
            "unresolved_count": len(draft.unresolved_items),
        }
