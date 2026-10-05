---
document_id: SOP-PAY-001
title: Payment Troubleshooting SOP
doc_type: sop
category: Payment
tags: payment, sop, troubleshooting, upi, card
created_date: 2023-11-01
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic content for demonstration only.

## Purpose

Standard operating procedure for support agents handling payment-related
complaints (failed transactions, duplicate charges, payment deducted but
order failed, payment reversed, payment pending).

## Steps

1. Confirm the order ID and payment method with the customer.
2. Check the internal payment ledger for the transaction status. Three
   possible states: `success`, `failed`, `pending_reconciliation`.
3. If `pending_reconciliation`, inform the customer that funds may show as
   deducted temporarily and reconciliation typically completes within
   30-60 minutes; do not initiate a manual refund yet.
4. If the transaction shows `failed` on our side but the customer has bank
   confirmation of a debit, check for an active payment gateway incident
   before manually processing a refund (see Emerging Issues / Incident
   log, e.g. INC-PAY-002).
5. If no active incident is found and reconciliation does not resolve
   within 24 hours, escalate to the Payments engineering on-call and
   initiate a manual refund per SOP-REF-003.
6. For duplicate charges, verify both transaction IDs are present in the
   ledger before refunding the duplicate amount.

## Escalation Criteria

Escalate immediately (do not wait) if:
- Complaint volume for a payment subcategory exceeds baseline by >100% in
  a single hour, OR
- The customer reports a transaction they did not authorize (treat as
  potential fraud, follow SOP-FRAUD-001 - not itself included in this demo
  KB, route to the fraud team).
