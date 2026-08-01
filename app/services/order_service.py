class OrderService:
    """
    Looking up order data. This class knows NOTHING about OpenAI, tools, or function calling — it's just a 
    normal business-logic class.
    This is the actual "real" code the LLM will
    trigger, but never touches directly.

    Uses a fake in-memory dict here for learning purposes — in a real
    project, this would query a database instead
    """
    def __init__(self):
        self._fake_orders = {
            "ORD1001": {"status": "Shipped", "eta": "2 days", "item": "Wireless Mouse"},
            "ORD1002": {"status": "Processing", "eta": "5 days", "item": "Mechanical Keyboard"},
            "ORD1003": {"status": "Delivered", "eta": None, "item": "USB-C Hub"},
        }


    def get_order_status(self, order_id: str) -> dict:
        """
        Returns order details, or a clear "not found" response if the
        order_id doesn't exist — the model needs to see SOMETHING to
        respond to, not a raw exception.
        """
        order = self._fake_orders.get(order_id.upper())
        if order is None:
            return {"error": f"No order found with ID '{order_id}'"}
        return {"order_id": order_id.upper(), **order}


order_service = OrderService()