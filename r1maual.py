def find_best_trading_path():
    """
    Find the highest-return trading path starting with 500,000 SeaShells,
    making exactly 5 trades, and ending with SeaShells.
    """
    # Define currencies and exchange rates
    currencies = ["Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells"]
    
    # Exchange rates matrix: [from][to]
    rates = {
        "Snowballs": {"Snowballs": 1, "Pizza's": 1.45, "Silicon Nuggets": 0.52, "SeaShells": 0.72},
        "Pizza's": {"Snowballs": 0.7, "Pizza's": 1, "Silicon Nuggets": 0.31, "SeaShells": 0.48},
        "Silicon Nuggets": {"Snowballs": 1.95, "Pizza's": 3.1, "Silicon Nuggets": 1, "SeaShells": 1.49},
        "SeaShells": {"Snowballs": 1.34, "Pizza's": 1.98, "Silicon Nuggets": 0.64, "SeaShells": 1}
    }
    
    initial_amount = 500000  # Starting with 500,000 SeaShells
    initial_currency = "SeaShells"
    final_currency = "SeaShells"  # Must end with SeaShells
    num_trades = 5
    
    # Variables to store the best path and maximum value
    best_path = []
    max_final_amount = 0
    paths_explored = 0
    
    # Recursive function to explore all possible trading paths
    def explore_trades(current_currency, amount, trades_made, path):
        nonlocal best_path, max_final_amount, paths_explored
        
        # Base case: we've made 5 trades
        if trades_made == num_trades:
            paths_explored += 1
            if current_currency == final_currency and amount > max_final_amount:
                max_final_amount = amount
                best_path = path.copy()
            return
        
        # If we're at the 4th trade, the 5th trade must go to SeaShells
        if trades_made == num_trades - 1:
            next_currency = final_currency
            new_amount = amount * rates[current_currency][next_currency]
            
            path.append((current_currency, next_currency, amount, new_amount))
            explore_trades(next_currency, new_amount, trades_made + 1, path)
            path.pop()  # Backtrack
            return
        
        # For trades 1-4, try all possible currencies except current one and SeaShells
        for next_currency in currencies:
            # Skip trading to the same currency (no point)
            if next_currency == current_currency:
                continue
            
            # Skip trading to SeaShells before the last trade
            if next_currency == final_currency and trades_made < num_trades - 1:
                continue
            
            # Calculate new amount after trade
            new_amount = amount * rates[current_currency][next_currency]
            
            # Add this trade to the path and continue exploring
            path.append((current_currency, next_currency, amount, new_amount))
            explore_trades(next_currency, new_amount, trades_made + 1, path)
            path.pop()  # Backtrack
    
    # Start the exploration
    explore_trades(initial_currency, initial_amount, 0, [])
    
    print(f"Total paths explored: {paths_explored}")
    return best_path, final_currency, max_final_amount, initial_amount

# Run the function and display results
best_path, final_currency, final_amount, initial_amount = find_best_trading_path()

print(f"\nStarting with {initial_amount} SeaShells")
print(f"Best trading path yields: {final_amount:.2f} {final_currency}")
print(f"Total earned: {final_amount - initial_amount:.2f} SeaShells")
print("\nTrading sequence:")
for i, (from_curr, to_curr, from_amount, to_amount) in enumerate(best_path):
    print(f"Trade {i+1}: {from_amount:.2f} {from_curr} → {to_amount:.2f} {to_curr}")

# Calculate the return as a percentage
return_percentage = (final_amount / initial_amount - 1) * 100
print(f"\nReturn on investment: {return_percentage:.2f}%")