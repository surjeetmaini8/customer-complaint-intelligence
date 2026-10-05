---
document_id: SOP-REF-003
title: Refund Standard Operating Procedure
doc_type: sop
category: Refund
tags: refund, sop, delayed, missing, partial
created_date: 2023-11-05
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic content for demonstration only.

## Purpose

Defines the standard timeline and process for handling refund requests and
refund-related complaints.

## Standard Refund Timeline

- Refund initiation: within 24 hours of return/cancellation approval.
- Refund to source (card/UPI/net banking): 3-7 business days, depending on
  the customer's bank.
- Refund to wallet: instant to 24 hours.

## Handling "Refund Delayed" Complaints

1. Confirm the refund initiation date in the system.
2. If within the standard timeline above, inform the customer of the
   expected completion date.
3. If beyond the standard timeline, check the refund transaction status
   with the payment gateway partner.
4. If the gateway confirms the refund was sent but the customer has not
   received it after 10 business days, escalate to Payments engineering
   for manual reconciliation.

## Handling "Refund Missing" / "Partial Refund" Complaints

1. Verify the original order amount against the refunded amount in the
   ledger.
2. If a partial refund was issued incorrectly (e.g. shipping fee not
   refunded when it should have been), issue a manual top-up refund for
   the difference after supervisor approval.
3. Document the reason for any partial refund clearly in the ticket notes
   so root-cause analysis can distinguish policy-correct partial refunds
   from system errors.

## Related Documents

- SOP-PAY-001 (Payment Troubleshooting SOP)
- INC-PAY-002 (UPI Payment Gateway Failure Spike)
