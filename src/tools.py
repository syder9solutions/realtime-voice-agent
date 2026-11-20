"""Deterministic, schema-validated business tools for the support-line agent."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.models import (
    BusinessHoursArgs,
    LookupOrderArgs,
    RescheduleDeliveryArgs,
    StrictModel,
)

ORDERS: dict[str, dict[str, str]] = {
    "ORD-123": {"status": "in transit", "eta": "tomorrow 14:00", "item": "wireless headphones"},
    "ORD-456": {"status": "processing", "eta": "in 2 days", "item": "mechanical keyboard"},
    "ORD-789": {"status": "delivered", "eta": "delivered yesterday", "item": "usb-c hub"},
}


class ToolError(ValueError):
    """Invalid or failed tool invocation."""


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        schema: type[StrictModel],
        handler: Callable[[BaseModel], str],
    ) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler

    def validate(self, args: dict[str, Any]) -> BaseModel:
        try:
            return self.schema.model_validate(args)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments: {exc.errors()}") from exc

    def call(self, args: dict[str, Any]) -> str:
        return self.handler(self.validate(args))

    def json_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, **self.schema.model_json_schema()}


def _lookup(args: LookupOrderArgs) -> str:
    order = ORDERS.get(args.order_id.upper())
    if order is None:
        return f"No order found for {args.order_id}."
    return (
        f"Order {args.order_id.upper()}: {order['item']}, status {order['status']}, "
        f"ETA {order['eta']}."
    )


def _reschedule(args: RescheduleDeliveryArgs) -> str:
    key = args.order_id.upper()
    if key not in ORDERS:
        return f"Cannot reschedule: order {args.order_id} not found."
    ORDERS[key] = {**ORDERS[key], "eta": args.new_date}
    return f"Order {key} delivery rescheduled to {args.new_date}."


def _hours(_args: BusinessHoursArgs) -> str:
    return "Support is available Monday to Friday, 09:00-18:00 local time."


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.register(
            Tool("lookup_order", "Look up an order's status and ETA.", LookupOrderArgs, _lookup)
        )
        self.register(
            Tool(
                "reschedule_delivery",
                "Change the delivery date of an existing order.",
                RescheduleDeliveryArgs,
                _reschedule,
            )
        )
        self.register(
            Tool("business_hours", "Return customer-support opening hours.", BusinessHoursArgs, _hours)
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        return tool.call(args)

    def describe(self) -> list[dict[str, Any]]:
        return [tool.json_schema() for tool in self._tools.values()]
