import torch

def RF_linear_data(num_obs=20000, num_products=3):
    """
    Generates data for a reduced form model with linear and quadratic price effects.
    Returns quantities with proper downward-sloping demand properties.
    """
    m_price = torch.full((num_products,), 0.5)
    m, j = num_obs, num_products
    
    # Generate base prices with some random variation
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    z = torch.rand(m, j, dtype=torch.float32) - 0.5
    p_jt = m_price + 0.3 * (z + u + torch.rand(m, j, dtype=torch.float32) - 0.5)
    
    # Ensure prices are positive
    p_jt = torch.clamp(p_jt, min=0.1)
    
    # Generate quantities with polynomial price effects
    beta_0 = 0.4  # intercept
    beta_1 = -0.5  # linear price effect
    beta_2 = -0.2  # quadratic price effect
    
    # Generate quantities - each product depends only on its own price
    quantities = beta_0 + beta_1 * p_jt + beta_2 * p_jt.pow(2) + 0.1 * u
    
    # Ensure quantities are positive
    quantities = torch.clamp(quantities, min=0.01)
    
    # Return only the first product's data for elasticity estimation
    return quantities[:, 0], torch.cat((p_jt, z[:, [0]]), dim=1), p_jt[:, 0], z[:, 0]

def RF_linear_stats(num_obs=20000, num_products=3):
    """
    Calculates true elasticity for the reduced form model.
    """
    beta_1 = -0.5
    beta_2 = -0.2
    mean_price = torch.tensor(0.5)
    
    # Calculate mean quantity at mean price
    mean_quantity = 0.4 + beta_1 * mean_price + beta_2 * mean_price.pow(2)
    
    # Own-price elasticity: (dQ/dP)*(P/Q)
    price_derivative = beta_1 + 2 * beta_2 * mean_price
    e_own = price_derivative * mean_price / mean_quantity
    
    return e_own.item()

def RF_quadratic_data(num_obs=20000, num_products=3):
    """
    Generates data for a reduced form model with higher order price effects.
    Each product's quantity depends only on its own price.
    """
    m_price = torch.full((num_products,), 0.5)
    m, j = num_obs, num_products
    
    # Generate base prices with some random variation
    u = torch.rand(m, j, dtype=torch.float32) - 0.5
    z = torch.rand(m, j, dtype=torch.float32) - 0.5
    p_jt = m_price + 0.3 * (z + u + torch.rand(m, j, dtype=torch.float32) - 0.5)
    
    # Ensure prices are positive
    p_jt = torch.clamp(p_jt, min=0.1)
    
    # Parameters for price effects
    beta_0 = 0.4
    beta_1 = -0.5
    beta_2 = -0.2
    beta_3 = -0.1  # cubic price effect
    
    # Generate quantities with cubic price effects
    quantities = beta_0 + beta_1 * p_jt + beta_2 * p_jt.pow(2) + beta_3 * p_jt.pow(3) + 0.1 * u
    
    # Ensure quantities are positive
    quantities = torch.clamp(quantities, min=0.01)
    
    # Return only the first product's data for elasticity estimation
    return quantities[:, 0], torch.cat((p_jt, z[:, [0]]), dim=1), p_jt[:, 0], z[:, 0]

def RF_quadratic_stats(num_obs=20000, num_products=3):
    """
    Calculates true elasticity for the quadratic model.
    """
    beta_1 = -0.5
    beta_2 = -0.2
    beta_3 = -0.1
    mean_price = torch.tensor(0.5)
    
    # Calculate mean quantity at mean price
    mean_quantity = 0.4 + beta_1 * mean_price + beta_2 * mean_price.pow(2) + beta_3 * mean_price.pow(3)
    
    # Own-price elasticity with cubic term
    price_derivative = beta_1 + 2 * beta_2 * mean_price + 3 * beta_3 * mean_price.pow(2)
    e_own = price_derivative * mean_price / mean_quantity
    
    return e_own.item()