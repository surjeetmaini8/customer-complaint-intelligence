---
document_id: KB-OTP-004
title: Troubleshooting - OTP Not Received
doc_type: troubleshooting
category: Technical Issue
tags: OTP, login, troubleshooting, faq
created_date: 2024-01-05
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic content for demonstration only.

## Common Causes of OTP Delivery Failure

1. **SMS gateway provider outage or latency** - the most common root cause
   for sudden, large-scale OTP failure spikes. Check the current status of
   the SMS gateway provider dashboard and recent incident reports (see
   INC-OTP-001) before troubleshooting individual accounts.
2. **Incorrect or outdated phone number on file** - verify the number
   registered to the account matches the customer's current number.
3. **Carrier-side spam filtering** - some telecom carriers filter
   automated SMS from unrecognized short codes.
4. **Device-side notification/SMS permission issues** - primarily affects
   Android devices running custom battery-optimization software that can
   delay SMS delivery to the OS.
5. **Poor network connectivity** on the customer's device.

## Recommended Troubleshooting Steps

1. Ask the customer to confirm the registered phone number is correct.
2. Ask the customer to check for network connectivity issues.
3. Check whether the issue is isolated to one customer or part of a
   broader spike (cross-reference the Emerging Issues dashboard).
4. If it is a broader spike, escalate immediately to the authentication
   engineering team rather than troubleshooting individual tickets.
5. Offer the customer an alternate verification method (email OTP or
   support-assisted verification) as a temporary workaround during known
   incidents.

## Escalation Criteria

Escalate to Authentication Engineering if:
- More than 20 OTP-failure complaints are received within a single hour, OR
- OTP failures are concentrated on a single platform/region combination.
