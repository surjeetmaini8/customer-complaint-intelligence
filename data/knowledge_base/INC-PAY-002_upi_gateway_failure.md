---
document_id: INC-PAY-002
title: UPI Payment Gateway Failure Spike - February 2024
doc_type: incident
category: Payment
tags: UPI, payment gateway, transaction failure, android, incident
created_date: 2024-02-14
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic, fictional incident report for
> demonstration purposes only.

## Summary

On 2024-02-14, between roughly 07:00 and 20:00, a significant spike in
failed UPI transactions was observed, concentrated among Android users in
the South India region. Customers reported that payments failed at
checkout, and in some cases funds were debited from the customer's bank
account without a corresponding successful order.

## Root Cause

The UPI payment gateway partner experienced degraded performance on their
transaction-status callback API during a high-traffic promotional period.
The callback delay caused our system to mark otherwise-successful payments
as "failed" after a 15-second timeout, even though the debit had already
been processed on the bank's side, producing "payment deducted but order
failed" complaints.

## Impact

- UPI transaction failure rate rose from a baseline of ~3% to over 40%
  during peak hours of the incident.
- A large volume of "payment deducted but order failed" and "failed
  transaction" complaints were logged, concentrated on Android/South India.
- Increased load on the Refunds team to reconcile deducted-but-unconfirmed
  payments.

## Resolution

1. Increased the transaction-status callback timeout from 15s to 45s.
2. Implemented an asynchronous reconciliation job that automatically
   matches bank-side debits to order status every 5 minutes and
   auto-completes or auto-refunds orders accordingly.
3. Added a customer-facing "payment processing" state to avoid premature
   "failed" messaging during gateway slowness.

## Related Documents

- SOP-PAY-001 (Payment Troubleshooting SOP)
- SOP-REF-003 (Refund SOP)
