-- Setup script for PostgreSQL test data
-- Run this manually in your smolval-test database before running the MCP comparison
-- This creates sample data that the read-only MCP servers can query

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Create categories table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    in_stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    total_amount DECIMAL(10,2),
    shipping_address TEXT
);

-- Create order_items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);

-- Insert sample categories
INSERT INTO categories (name, description) VALUES
('Electronics', 'Electronic devices and gadgets'),
('Books', 'Physical and digital books'),
('Clothing', 'Apparel and accessories'),
('Home & Garden', 'Home improvement and gardening supplies'),
('Sports', 'Sports equipment and gear');

-- Insert sample products
INSERT INTO products (name, description, price, category_id, in_stock, metadata) VALUES
('Laptop Pro 15"', 'High-performance laptop for professionals', 1299.99, 1, 25, '{"brand": "TechCorp", "warranty": "2 years"}'),
('Wireless Mouse', 'Ergonomic wireless mouse with precision tracking', 29.99, 1, 150, '{"brand": "TechCorp", "color": "black"}'),
('The Great Novel', 'Bestselling fiction novel of the year', 15.99, 2, 75, '{"author": "Jane Author", "pages": 324}'),
('Programming Guide', 'Complete guide to modern programming', 49.99, 2, 30, '{"author": "Code Master", "edition": "5th"}'),
('Cotton T-Shirt', 'Comfortable cotton t-shirt in various colors', 19.99, 3, 200, '{"material": "100% cotton", "sizes": ["S", "M", "L", "XL"]}'),
('Denim Jeans', 'Classic fit denim jeans', 79.99, 3, 100, '{"material": "denim", "fit": "classic"}'),
('Garden Hose 50ft', 'Durable garden hose for outdoor use', 34.99, 4, 45, '{"length": "50 feet", "material": "rubber"}'),
('Flower Seeds Mix', 'Variety pack of flower seeds for spring planting', 12.99, 4, 80, '{"season": "spring", "variety": "mixed"}'),
('Tennis Racket', 'Professional grade tennis racket', 89.99, 5, 20, '{"weight": "300g", "grip_size": "4 1/4"}'),
('Basketball', 'Official size basketball for indoor/outdoor play', 24.99, 5, 60, '{"size": "official", "material": "synthetic leather"}');

-- Insert sample customers
INSERT INTO customers (first_name, last_name, email, phone, address) VALUES
('John', 'Doe', 'john.doe@email.com', '555-0101', '123 Main St, Anytown, USA'),
('Jane', 'Smith', 'jane.smith@email.com', '555-0102', '456 Oak Ave, Other City, USA'),
('Bob', 'Johnson', 'bob.j@email.com', '555-0103', '789 Pine Rd, Another Town, USA'),
('Alice', 'Brown', 'alice.brown@email.com', '555-0104', '321 Elm St, Some City, USA'),
('Charlie', 'Wilson', 'charlie.w@email.com', '555-0105', '654 Maple Dr, That Place, USA');

-- Insert sample orders
INSERT INTO orders (customer_id, order_date, status, total_amount, shipping_address) VALUES
(1, '2024-01-15 10:30:00', 'completed', 1329.98, '123 Main St, Anytown, USA'),
(2, '2024-01-16 14:15:00', 'shipped', 65.98, '456 Oak Ave, Other City, USA'),
(3, '2024-01-17 09:45:00', 'pending', 104.98, '789 Pine Rd, Another Town, USA'),
(4, '2024-01-18 16:20:00', 'completed', 47.98, '321 Elm St, Some City, USA'),
(5, '2024-01-19 11:10:00', 'processing', 114.98, '654 Maple Dr, That Place, USA');

-- Insert sample order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Order 1: Laptop + Mouse
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
-- Order 2: Books
(2, 3, 2, 15.99),
(2, 4, 1, 49.99),
-- Order 3: Clothing
(3, 5, 3, 19.99),
(3, 6, 1, 79.99),
-- Order 4: Garden supplies
(4, 7, 1, 34.99),
(4, 8, 1, 12.99),
-- Order 5: Sports equipment
(5, 9, 1, 89.99),
(5, 10, 1, 24.99);

-- Create some indexes for testing
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);

-- Create a view for testing
CREATE VIEW product_summary AS
SELECT 
    p.id,
    p.name,
    p.price,
    c.name as category_name,
    p.in_stock,
    CASE 
        WHEN p.in_stock > 100 THEN 'High Stock'
        WHEN p.in_stock > 20 THEN 'Medium Stock'
        ELSE 'Low Stock'
    END as stock_level
FROM products p
JOIN categories c ON p.category_id = c.id;

-- Add some additional test data with edge cases
INSERT INTO products (name, description, price, category_id, in_stock, metadata) VALUES
('Special Item', 'Item with special characters: café, naïve, résumé', 99.99, 1, 5, '{"unicode": "test", "price_history": [89.99, 94.99, 99.99]}'),
('Expensive Item', 'Very expensive luxury item', 9999.99, 3, 1, '{"luxury": true, "limited_edition": true}'),
('Free Sample', 'Free promotional item', 0.00, 4, 1000, '{"promotional": true, "cost": 0}');

-- Print confirmation
SELECT 
    'Setup complete!' as message,
    (SELECT COUNT(*) FROM categories) as categories_count,
    (SELECT COUNT(*) FROM products) as products_count,
    (SELECT COUNT(*) FROM customers) as customers_count,
    (SELECT COUNT(*) FROM orders) as orders_count,
    (SELECT COUNT(*) FROM order_items) as order_items_count;