-- =============================================================================
-- ROLLINGMRP DATABASE SCHEMA
-- Compatible with SQLite and ANSI SQL / PostgreSQL
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- 1. SUPPLIER
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    reliability_score REAL NOT NULL CHECK (reliability_score >= 0.0 AND reliability_score <= 1.0),
    quoted_lead_time_days INTEGER NOT NULL CHECK (quoted_lead_time_days >= 0),
    actual_lead_time_mean REAL NOT NULL CHECK (actual_lead_time_mean >= 0.0),
    actual_lead_time_stddev REAL NOT NULL CHECK (actual_lead_time_stddev >= 0.0),
    fill_rate REAL NOT NULL CHECK (fill_rate >= 0.0 AND fill_rate <= 1.0),
    quality_reject_rate REAL NOT NULL CHECK (quality_reject_rate >= 0.0 AND quality_reject_rate <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- 2. ITEM
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS items (
    item_id VARCHAR(50) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('raw_material', 'component', 'subassembly', 'finished_good')),
    unit_of_measure VARCHAR(20) NOT NULL,
    standard_cost REAL NOT NULL CHECK (standard_cost >= 0.0),
    shelf_life_days INTEGER CHECK (shelf_life_days IS NULL OR shelf_life_days > 0),
    lot_sizing_rule VARCHAR(50) NOT NULL CHECK (lot_sizing_rule IN ('L4L', 'EOQ', 'POQ', 'WAGNER_WHITIN', 'SILVER_MEAL', 'LUC', 'FIXED_PERIOD')),
    safety_stock REAL NOT NULL DEFAULT 0.0 CHECK (safety_stock >= 0.0),
    lead_time_days INTEGER NOT NULL CHECK (lead_time_days >= 0),
    supplier_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE SET NULL
);

-- -----------------------------------------------------------------------------
-- 3. BILL OF MATERIALS (BOM)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bom (
    bom_id VARCHAR(50) PRIMARY KEY,
    parent_item_id VARCHAR(50) NOT NULL,
    component_item_id VARCHAR(50) NOT NULL,
    quantity_per REAL NOT NULL CHECK (quantity_per > 0.0),
    scrap_factor REAL NOT NULL DEFAULT 0.0 CHECK (scrap_factor >= 0.0 AND scrap_factor < 1.0),
    effective_date TEXT,
    obsolete_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_item_id) REFERENCES items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (component_item_id) REFERENCES items(item_id) ON DELETE RESTRICT,
    CONSTRAINT bom_unique_parent_component UNIQUE (parent_item_id, component_item_id)
);

-- -----------------------------------------------------------------------------
-- 4. INVENTORY
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id VARCHAR(50) NOT NULL,
    location_id VARCHAR(50) NOT NULL,
    quantity_on_hand REAL NOT NULL DEFAULT 0.0 CHECK (quantity_on_hand >= 0.0),
    quantity_allocated REAL NOT NULL DEFAULT 0.0 CHECK (quantity_allocated >= 0.0),
    quantity_on_order REAL NOT NULL DEFAULT 0.0 CHECK (quantity_on_order >= 0.0),
    expiration_date TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE,
    CONSTRAINT inv_unique_item_location UNIQUE (item_id, location_id)
);

-- -----------------------------------------------------------------------------
-- 5. WORK CENTER
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS work_centers (
    work_center_id VARCHAR(50) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    capacity_hours_per_week REAL NOT NULL CHECK (capacity_hours_per_week >= 0.0),
    efficiency_factor REAL NOT NULL DEFAULT 1.0 CHECK (efficiency_factor > 0.0 AND efficiency_factor <= 2.0),
    calendar_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- 6. ROUTING
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS routings (
    routing_id VARCHAR(50) PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    operation_sequence INTEGER NOT NULL CHECK (operation_sequence > 0),
    work_center_id VARCHAR(50) NOT NULL,
    setup_time_hours REAL NOT NULL DEFAULT 0.0 CHECK (setup_time_hours >= 0.0),
    run_time_hours_per_unit REAL NOT NULL DEFAULT 0.0 CHECK (run_time_hours_per_unit >= 0.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (work_center_id) REFERENCES work_centers(work_center_id) ON DELETE RESTRICT,
    CONSTRAINT routing_item_sequence_unique UNIQUE (item_id, operation_sequence)
);

-- -----------------------------------------------------------------------------
-- 7. MASTER PRODUCTION SCHEDULE (MPS)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mps (
    mps_id VARCHAR(50) PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    period_number INTEGER NOT NULL CHECK (period_number >= 1),
    quantity REAL NOT NULL CHECK (quantity >= 0.0),
    demand_type VARCHAR(50) NOT NULL CHECK (demand_type IN ('actual', 'forecast', 'firm_planned')),
    source VARCHAR(50) NOT NULL CHECK (source IN ('customer_order', 'forecast', 'manual')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE,
    CONSTRAINT mps_item_period_type UNIQUE (item_id, period_number, demand_type)
);

-- -----------------------------------------------------------------------------
-- 8. PLANNED ORDER
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS planned_orders (
    order_id VARCHAR(50) PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    quantity REAL NOT NULL CHECK (quantity > 0.0),
    release_date TEXT NOT NULL,
    due_date TEXT NOT NULL,
    lot_sizing_rule_applied VARCHAR(50) NOT NULL,
    pegging_chain TEXT, -- JSON structure representing order provenance & upstream demand links
    status VARCHAR(50) NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'released', 'firm', 'cancelled')),
    nervousness_flag BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 9. EXCEPTION MESSAGE
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exception_messages (
    message_id VARCHAR(50) PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    message_type VARCHAR(50) NOT NULL CHECK (message_type IN ('expedite', 'defer', 'cancel', 'new_order', 'safety_stock')),
    current_due_date TEXT,
    suggested_due_date TEXT,
    quantity_impact REAL DEFAULT 0.0,
    dollar_impact REAL DEFAULT 0.0,
    priority_score REAL DEFAULT 0.0,
    status VARCHAR(50) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'resolved')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- INDEXES
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_items_type ON items(type);
CREATE INDEX IF NOT EXISTS idx_items_supplier ON items(supplier_id);
CREATE INDEX IF NOT EXISTS idx_bom_parent ON bom(parent_item_id);
CREATE INDEX IF NOT EXISTS idx_bom_component ON bom(component_item_id);
CREATE INDEX IF NOT EXISTS idx_inventory_item ON inventory(item_id);
CREATE INDEX IF NOT EXISTS idx_routings_item ON routings(item_id);
CREATE INDEX IF NOT EXISTS idx_routings_work_center ON routings(work_center_id);
CREATE INDEX IF NOT EXISTS idx_mps_item_period ON mps(item_id, period_number);
CREATE INDEX IF NOT EXISTS idx_planned_orders_item ON planned_orders(item_id);
CREATE INDEX IF NOT EXISTS idx_planned_orders_dates ON planned_orders(release_date, due_date);
CREATE INDEX IF NOT EXISTS idx_exceptions_item_status ON exception_messages(item_id, status);

