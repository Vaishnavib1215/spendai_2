const mysql = require('mysql2/promise');

const DB_NAME = process.env.DB_NAME || 'spendai';

let pool;

async function ensureDatabase() {
  const conn = await mysql.createConnection({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '3306'),
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
  });
  await conn.query(`CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci`);
  await conn.end();

  pool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '3306'),
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: DB_NAME,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
  });
}

async function runAsync(sql, params = []) {
  const [result] = await pool.execute(sql, params);
  return { lastID: result.insertId, changes: result.affectedRows };
}

async function getAsync(sql, params = []) {
  const [rows] = await pool.execute(sql, params);
  return rows[0] || null;
}

async function allAsync(sql, params = []) {
  const [rows] = await pool.execute(sql, params);
  return rows;
}

async function initSchema() {
  await ensureDatabase();

  const statements = [
    `CREATE TABLE IF NOT EXISTS Users (
      user_id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )`,
    `CREATE TABLE IF NOT EXISTS Categories (
      category_id INT AUTO_INCREMENT PRIMARY KEY,
      category_name VARCHAR(255) NOT NULL,
      category_type ENUM('Need','Want') NOT NULL
    )`,
    `CREATE TABLE IF NOT EXISTS Transactions (
      transaction_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id INT NOT NULL,
      amount DOUBLE NOT NULL,
      category_id INT,
      transaction_date VARCHAR(20) NOT NULL,
      payment_mode VARCHAR(100),
      description TEXT,
      FOREIGN KEY (user_id) REFERENCES Users(user_id),
      FOREIGN KEY (category_id) REFERENCES Categories(category_id)
    )`,
    `CREATE TABLE IF NOT EXISTS Budgets (
      budget_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id INT UNIQUE NOT NULL,
      weekly_limit DOUBLE,
      monthly_limit DOUBLE,
      FOREIGN KEY (user_id) REFERENCES Users(user_id)
    )`,
    `CREATE TABLE IF NOT EXISTS Predictions (
      prediction_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id INT NOT NULL,
      week_number INT,
      predicted_spending DOUBLE,
      overspend_risk INT,
      reason TEXT,
      cluster_label VARCHAR(100),
      recommended_budget DOUBLE,
      predicted_monthly DOUBLE,
      result_json LONGTEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES Users(user_id)
    )`,
  ];

  for (const sql of statements) {
    await runAsync(sql);
  }

  // Create index (ignore error if already exists)
  try {
    await pool.query('CREATE INDEX idx_user_date ON Transactions(user_id, transaction_date)');
  } catch (err) {
    if (err.code !== 'ER_DUP_KEYNAME') throw err;
  }

  // Migrate older databases: add columns introduced after first release
  for (const col of ['predicted_monthly DOUBLE', 'result_json LONGTEXT']) {
    try {
      await pool.query(`ALTER TABLE Predictions ADD COLUMN ${col}`);
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') throw err;
    }
  }

  // Seed categories
  const count = await getAsync('SELECT COUNT(*) as c FROM Categories');
  if (count.c === 0) {
    const seeds = [
      ['Food', 'Need'], ['Transport', 'Need'], ['Shopping', 'Want'],
      ['Entertainment', 'Want'], ['Healthcare', 'Need'], ['Education', 'Need'],
      ['Utilities', 'Need'], ['Travel', 'Want'],
    ];
    for (const [name, type] of seeds) {
      await runAsync('INSERT INTO Categories (category_name, category_type) VALUES (?, ?)', [name, type]);
    }
  }
}

module.exports = { getPool: () => pool, runAsync, getAsync, allAsync, initSchema };
