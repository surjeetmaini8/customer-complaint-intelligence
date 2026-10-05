---
document_id: SOP-ACC-002
title: Account Recovery and Lockout SOP
doc_type: sop
category: Account
tags: account, login, password, locked, recovery
created_date: 2023-09-15
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic content for demonstration only.

## Purpose

Defines the process for handling login issues, password resets, and
account lockouts.

## Login Issues

1. Confirm the customer is using the correct registered email/phone.
2. Check for an active authentication service incident (see Emerging
   Issues) before assuming user error - login issues can spike due to
   backend authentication problems, not just individual mistakes.
3. Guide the customer through a password reset if credentials appear
   incorrect.

## Password Reset Not Working

1. Confirm the reset email/SMS was sent to the correct, currently
   registered contact method.
2. Ask the customer to check spam/junk folders.
3. If the reset link is expired, generate a new one - links expire after
   30 minutes for security.
4. If resets consistently fail across multiple customers, escalate to
   engineering; this may indicate a broader delivery issue similar to
   OTP failures (see KB-OTP-004).

## Account Locked

1. Accounts lock automatically after 5 failed login attempts within 15
   minutes, as a security measure.
2. Verify the customer's identity using two independent data points
   (e.g. registered email + last order ID) before unlocking.
3. Unlock the account and advise the customer to reset their password
   immediately after unlocking.
4. If the customer reports they did NOT attempt to log in (i.e. did not
   trigger the lockout themselves), treat this as a potential account
   security/fraud signal and flag for the fraud/security team.
