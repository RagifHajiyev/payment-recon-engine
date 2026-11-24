-- Reporting views over the reconciliation results (SQLite).

DROP VIEW IF EXISTS v_daily_breaks;
CREATE VIEW v_daily_breaks AS
SELECT substr(ts, 1, 10)               AS day,
       issue,
       COUNT(*)                        AS records,
       ROUND(SUM(break_value_azn), 2)  AS value_azn
FROM reconciliation
WHERE issue <> 'matched'
GROUP BY day, issue;

DROP VIEW IF EXISTS v_partner_breaks;
CREATE VIEW v_partner_breaks AS
SELECT COALESCE(partner, 'unknown')    AS partner,
       issue,
       COUNT(*)                        AS records,
       ROUND(SUM(break_value_azn), 2)  AS value_azn
FROM reconciliation
WHERE issue <> 'matched'
GROUP BY partner, issue;

DROP VIEW IF EXISTS v_unmatched_value;
CREATE VIEW v_unmatched_value AS
SELECT issue,
       COUNT(*)                                            AS records,
       ROUND(SUM(break_value_azn), 2)                      AS value_azn,
       ROUND(100.0 * SUM(break_value_azn) /
             (SELECT SUM(break_value_azn) FROM reconciliation), 2) AS pct_of_break_value
FROM reconciliation
WHERE issue <> 'matched'
GROUP BY issue
ORDER BY value_azn DESC;

DROP VIEW IF EXISTS v_fee_flags;
CREATE VIEW v_fee_flags AS
SELECT partner, month, turnover_azn, expected_rate, implied_rate,
       expected_fee, fee_charged, fee_diff, fee_diff_pct
FROM fee_check
WHERE flagged = 1
ORDER BY ABS(fee_diff) DESC;

DROP VIEW IF EXISTS v_variance_by_partner_month;
CREATE VIEW v_variance_by_partner_month AS
SELECT partner, month,
       ROUND(SUM(expected_fee), 2) AS expected_fee,
       ROUND(SUM(fee_charged), 2)  AS fee_charged,
       ROUND(SUM(fee_diff), 2)     AS variance
FROM fee_check
GROUP BY partner, month
ORDER BY partner, month;
