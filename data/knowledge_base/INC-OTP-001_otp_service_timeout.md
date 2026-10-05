---
document_id: INC-OTP-001
title: OTP Service Timeout Incident - March 2024
doc_type: incident
category: Technical Issue
tags: OTP, authentication, timeout, android, incident
created_date: 2024-03-10
---

> DEMO KNOWLEDGE BASE DOCUMENT - this is a synthetic, fictional incident
> report created for demonstration purposes only. It does not describe a
> real company, system, or event.

## Summary

On 2024-03-10, between approximately 09:00 and 15:00, the OTP (One-Time
Password) delivery service experienced a sustained timeout condition when
communicating with the third-party SMS gateway provider. Customers on the
Android platform, predominantly in the North India region, were unable to
receive OTP codes required for login and checkout verification.

## Root Cause

Engineering traced the issue to an expired TLS certificate on the SMS
gateway provider's inbound API endpoint, which caused the authentication
service to time out after 30 seconds on every OTP dispatch request. The
authentication service did not have a fallback SMS provider configured for
this traffic corridor, so all Android-originated OTP requests routed
through the affected gateway failed silently from the user's perspective.

## Impact

- OTP delivery success rate dropped from a baseline of ~98% to under 15%
  during the incident window.
- Elevated complaint volume for the "OTP failure" subcategory, concentrated
  on Android devices in North India.
- Login and checkout completion rates dropped sharply for affected users.

## Resolution

1. Engineering renewed the expired TLS certificate at 14:50.
2. A secondary SMS gateway provider was configured as an automatic
   failover for future incidents.
3. Added active monitoring + alerting on OTP delivery success rate with a
   95% threshold.

## Related Documents

- KB-OTP-004 (Troubleshooting: OTP Not Received)
- SOP-ACC-002 (Account Recovery SOP)
