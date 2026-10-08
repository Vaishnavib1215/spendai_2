const express = require('express');
const axios = require('axios');
const { runAsync, getAsync, allAsync } = require('../db');
const { verifyToken } = require('../auth');

const router = express.Router();
router.use(verifyToken);

const ML_SERVER = process.env.ML_SERVER_URL || 'http://localhost:5001';

// ── date helpers (all dates are 'YYYY-MM-DD' strings, which sort correctly) ──
const pad = (n) => String(n).padStart(2, '0');
const fmt = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

function periodStarts(now = new Date()) {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const weekStart = new Date(today);
  weekStart.setDate(today.getDate() - ((today.getDay() + 6) % 7)); // Monday
  const monthStart = new Date(today.getFullYear(), today.getMonth(), 1);
  return { today: fmt(today), weekStart: fmt(weekStart), monthStart: fmt(monthStart) };
}

const num = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? 0 : Number(v));

// GET /api/analysis/run
router.get('/run', async (req, res) => {
  try {
    const userId = req.user.userId;
    const transactions = await allAsync(`
      SELECT t.transaction_id, t.amount,
             DATE_FORMAT(t.transaction_date, '%Y-%m-%d') AS transaction_date, t.payment_mode,
             c.category_name, c.category_type
      FROM Transactions t
      LEFT JOIN Categories c ON t.category_id = c.category_id
      WHERE t.user_id = ?
      ORDER BY transaction_date ASC
    `, [userId]);

    if (transactions.length < 3)
      return res.status(422).json({ error: 'Need at least 3 transactions to run analysis' });

    const budget = await getAsync('SELECT * FROM Budgets WHERE user_id = ?', [userId]);

    const mlRes = await axios.post(`${ML_SERVER}/analyze`, {
      transactions,
      weekly_limit: budget ? budget.weekly_limit : null,
      today: periodStarts().today,
    }, { timeout: 30000 });
    const result = mlRes.data;

    await runAsync(`
      INSERT INTO Predictions
        (user_id, week_number, predicted_spending, predicted_monthly, overspend_risk, reason,
         cluster_label, recommended_budget, result_json)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [userId, result.week_number, result.predicted_spending, result.predicted_monthly ?? 0,
        result.overspend_risk ? 1 : 0, result.reason, result.cluster_label,
        result.recommended_budget, JSON.stringify(result)]);

    res.json(result);
  } catch (err) {
    if (err.code === 'ECONNREFUSED')
      return res.status(503).json({ error: 'ML server is not running. Start it with: python ml-server/app.py' });
    const detail = err.response?.data?.error;
    res.status(500).json({ error: detail || err.message });
  }
});

// GET /api/analysis/history
router.get('/history', async (req, res) => {
  try {
    const rows = await allAsync(
      'SELECT * FROM Predictions WHERE user_id = ? ORDER BY created_at DESC, prediction_id DESC LIMIT 10',
      [req.user.userId]
    );
    // Expand the stored full result so the dashboard can render everything after a reload
    res.json(rows.map((r) => {
      let full = {};
      try { full = r.result_json ? JSON.parse(r.result_json) : {}; } catch { /* ignore */ }
      const { result_json, ...rest } = r;
      return { ...rest, ...full, overspend_risk: !!r.overspend_risk };
    }));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// GET /api/analysis/summary
router.get('/summary', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { today, weekStart, monthStart } = periodStarts();

    // transaction_date is stored as 'YYYY-MM-DD' text, so compare as text against
    // text boundaries (the old DATE_SUB(CURDATE()) comparison mixed text and DATE
    // types, which MySQL evaluates numerically and returned 0).
    const [totalSpending, weeklySpending, monthlySpending, categoryBreakdown,
           weeklyTrend, monthlyTrend, budget, dayTypeSplit] = await Promise.all([
      getAsync('SELECT SUM(amount) AS total, COUNT(*) AS count FROM Transactions WHERE user_id = ?', [userId]),
      getAsync('SELECT SUM(amount) AS total FROM Transactions WHERE user_id = ? AND transaction_date >= ? AND transaction_date <= ?',
        [userId, weekStart, today]),
      getAsync('SELECT SUM(amount) AS total FROM Transactions WHERE user_id = ? AND transaction_date >= ? AND transaction_date <= ?',
        [userId, monthStart, today]),
      allAsync(`SELECT c.category_name, c.category_type, SUM(t.amount) AS total
        FROM Transactions t LEFT JOIN Categories c ON t.category_id = c.category_id
        WHERE t.user_id = ? GROUP BY c.category_name, c.category_type ORDER BY total DESC`, [userId]),
      allAsync(`SELECT DATE_FORMAT(STR_TO_DATE(transaction_date, '%Y-%m-%d'), '%x-W%v') AS week, SUM(amount) AS total
        FROM Transactions WHERE user_id = ? GROUP BY week ORDER BY week DESC LIMIT 8`, [userId]),
      allAsync(`SELECT DATE_FORMAT(STR_TO_DATE(transaction_date, '%Y-%m-%d'), '%Y-%m') AS month, SUM(amount) AS total
        FROM Transactions WHERE user_id = ? GROUP BY month ORDER BY month DESC LIMIT 12`, [userId]),
      getAsync('SELECT * FROM Budgets WHERE user_id = ?', [userId]),
      getAsync(`SELECT
        SUM(CASE WHEN DAYOFWEEK(STR_TO_DATE(transaction_date, '%Y-%m-%d')) IN (1, 7) THEN amount ELSE 0 END) AS weekend,
        SUM(CASE WHEN DAYOFWEEK(STR_TO_DATE(transaction_date, '%Y-%m-%d')) NOT IN (1, 7) THEN amount ELSE 0 END) AS weekday
        FROM Transactions WHERE user_id = ?`, [userId]),
    ]);

    res.json({
      totalSpending: num(totalSpending?.total),
      totalTransactions: num(totalSpending?.count),
      weeklySpending: num(weeklySpending?.total),
      monthlySpending: num(monthlySpending?.total),
      categoryBreakdown: categoryBreakdown.map((c) => ({ ...c, total: num(c.total) })),
      weeklyTrend: weeklyTrend.map((w) => ({ ...w, total: num(w.total) })).reverse(),
      monthlyTrend: monthlyTrend.map((m) => ({ ...m, total: num(m.total) })).reverse(),
      budget: budget
        ? { ...budget, weekly_limit: num(budget.weekly_limit), monthly_limit: num(budget.monthly_limit) }
        : { weekly_limit: 5000, monthly_limit: 20000 },
      dayTypeSplit: { weekend: num(dayTypeSplit?.weekend), weekday: num(dayTypeSplit?.weekday) },
      period: { today, weekStart, monthStart },
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// PUT /api/analysis/budget
router.put('/budget', async (req, res) => {
  try {
    const weekly = parseFloat(req.body.weekly_limit);
    const monthly = parseFloat(req.body.monthly_limit);
    if (!(weekly >= 0) || !(monthly >= 0))
      return res.status(400).json({ error: 'weekly_limit and monthly_limit must be non-negative numbers' });
    // Upsert: users created before a Budgets row existed would otherwise be silently skipped
    await runAsync(
      `INSERT INTO Budgets (user_id, weekly_limit, monthly_limit) VALUES (?, ?, ?)
       ON DUPLICATE KEY UPDATE weekly_limit = VALUES(weekly_limit), monthly_limit = VALUES(monthly_limit)`,
      [req.user.userId, weekly, monthly]
    );
    res.json({ success: true });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

module.exports = router;
