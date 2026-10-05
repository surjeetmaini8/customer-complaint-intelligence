---
document_id: KB-APP-007
title: App Performance Troubleshooting FAQ
doc_type: faq
category: Technical Issue
tags: performance, slow, crash, app, faq
created_date: 2023-12-10
---

> DEMO KNOWLEDGE BASE DOCUMENT - synthetic content for demonstration only.

## Q: The app is very slow or unresponsive. What should I check?

A: Ask the customer to:
1. Update to the latest app version.
2. Clear the app cache (Settings > Apps > [App Name] > Storage > Clear
   Cache) - this does not delete account data.
3. Restart their device.
4. Check available device storage; less than 500MB free space commonly
   causes slowdowns.

## Q: The app crashes when I open a specific screen.

A: Check the Known Bugs log (e.g. BUG-APP-011) for an existing, matching
issue before filing a new bug report. If none match, collect the device
model, OS version, and app version, and escalate to the mobile
engineering team.

## Q: Pages are not loading on the website.

A: This is usually either a client-side network issue or a backend
API/performance issue. Ask the customer to try a different network/browser
first. If multiple customers report this simultaneously, treat it as a
potential backend incident and check system status before troubleshooting
individually.
