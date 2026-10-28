# Tính chi phí tồn kho, mô phỏng hàng dư/thiếu và tìm lượng đặt tối ưu bằng Grid Search.
# src/inventory.py
# Nhiệm vụ: quyết định lượng hàng cần đặt.

def optimize_order(current_inventory, predicted_demand, costs):
    # 1. Tạo danh sách các lượng đặt hàng Q.
    # 2. Tính chi phí của từng Q.
    # 3. Chọn Q có chi phí thấp nhất.
    # 4. Trả về Q tối ưu và chi phí dự kiến.
    pass











def calculate_inventory_cost(
    current_inventory,
    order_quantity,
    predicted_demand,
    holding_cost,
    stockout_cost
):
    # Tổng hàng sẵn có trong kỳ.
    available = current_inventory + order_quantity

    # Hàng còn dư sau khi đáp ứng nhu cầu.
    ending_inventory = max(available - predicted_demand, 0)

    # Nhu cầu không được đáp ứng.
    unmet_demand = max(predicted_demand - available, 0)

    # Tổng chi phí của phương án đặt hàng.
    total_cost = (
        holding_cost * ending_inventory
        + stockout_cost * unmet_demand
    )

    return total_cost


def grid_search_order(
    current_inventory,
    predicted_demand,
    holding_cost,
    stockout_cost,
    max_order_quantity=100
):
    best_order = None
    best_cost = float("inf")

    # Thử lần lượt các lượng đặt hàng từ 0 đến giới hạn.
    for order_quantity in range(max_order_quantity + 1):
        cost = calculate_inventory_cost(
            current_inventory=current_inventory,
            order_quantity=order_quantity,
            predicted_demand=predicted_demand,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost
        )

        # Ghi nhận phương án có chi phí thấp nhất.
        if cost < best_cost:
            best_cost = cost
            best_order = order_quantity

    return best_order, best_cost


# Ví dụ sử dụng
order, cost = grid_search_order(
    current_inventory=20,
    predicted_demand=35,
    holding_cost=2,
    stockout_cost=10,
    max_order_quantity=100
)

print("Recommended order:", order)
print("Estimated cost:", cost)
