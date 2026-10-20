# Relay Technologies: business context

Relay Technologies is a **fictional** company. It is OpsPilot's first customer
scenario. All Relay data in this repository is synthetic.

> This document describes the business, not the data's findings. It is safe to
> give to an analysis agent as background. It deliberately contains no
> information about the problems planted in the synthetic data, and no
> financial or performance figures. Any metric must be computed from the data.

## Product: RelayOS

RelayOS is a cloud-based warehouse fulfillment operations platform. It lets
customers:

- connect e-commerce storefronts and ERP systems;
- import and continuously sync inventory;
- configure warehouse picking, packing and fulfillment rules;
- manage order processing from order capture to shipment;
- monitor shipping performance (carrier SLAs, on-time shipment, exceptions).

RelayOS sits between a customer's sales channels and ERP on one side and their
warehouse floor and carriers on the other. It is only useful once data flows
reliably in both directions, which is why integration dominates onboarding.

## Customers

Two customer types:

- **E-commerce retailers** that run their own warehouses.
- **Third-party logistics (3PL) companies** that fulfill orders for many
  merchants, often across several sites.

Customers are grouped into segments that drive pricing, onboarding approach and
support model:

| Segment | Typical profile | Onboarding model |
|---|---|---|
| SMB | Single warehouse, a storefront platform and light ERP or accounting tool | Mostly self-serve with guided setup |
| Mid-Market | Several warehouses, an established mid-market ERP | Assigned onboarding specialist |
| Enterprise | Multi-site retailers and large 3PLs with complex, customized ERPs and IT security reviews | Dedicated implementation team and solutions engineers |

Other attributes recorded per account: plan tier, region (North America, EMEA,
APAC, LATAM) and acquisition channel (self-serve, inbound, outbound sales,
partner, referral).

## Business model

- **Subscription SaaS**, billed annually or monthly, on three plan tiers:
  Starter, Professional and Enterprise. Pricing scales with warehouses and order
  volume.
- **Value starts at go-live.** A customer gets no operational benefit until live
  orders run through RelayOS. Slow or failed onboarding delays value, makes
  customers less likely to expand, and raises the risk of early churn.
- **Onboarding is a cost center.** Implementation specialists, solutions
  engineers and support staff are the main costs of getting a customer live.
  Prolonged onboarding consumes more of all three.

## Onboarding process

Every customer goes through five stages, in order:

| # | Stage (data name) | What happens | Typical owners |
|---|---|---|---|
| 1 | Account setup / warehouse provisioning (`warehouse_provisioning`) | Tenant created; warehouses, locations, zones and users set up | Onboarding specialist, customer admin |
| 2 | ERP and inventory data integration (`erp_integration`) | ERP and storefront connectors configured; initial inventory import; ongoing sync verified | Solutions engineer, customer IT |
| 3 | Fulfillment workflow configuration (`fulfillment_configuration`) | Picking, packing, shipping and order-routing rules configured | Onboarding specialist, warehouse operations lead |
| 4 | Testing and validation (`testing_validation`) | Test orders run end to end; inventory and fulfillment results checked | Customer operations, onboarding specialist |
| 5 | Go-live (`go_live`) | Live orders processed in RelayOS | Customer operations, support |

**Success criteria**

- **Onboarding complete:** go-live is completed.
- **On target:** go-live within the segment's target time-to-value from signup:
  SMB 21 days, Mid-Market 40 days, Enterprise 75 days.
- **Abandoned:** customer success flags an onboarding as abandoned after 30
  days without onboarding activity.
- **In progress:** neither yet. Recent journeys may still complete or be
  abandoned, so they are not counted as failures.

## Key operational metrics

Leadership tracks onboarding health with these metrics. Each one must be
computed from the data with an explicit definition, never estimated.

| Metric | Definition | Watch for |
|---|---|---|
| Time-to-value | Signup to go-live completed | Segment, cohort and trend differences |
| On-target rate | Share of completed journeys that go live within target | Declines by segment or cohort |
| Stage duration | `stage_started` to `stage_completed` per stage | Stages getting slower, long tails |
| Stage conversion / abandonment | Share of customers entering a stage who complete or abandon it | Shifts at a particular stage |
| Funnel status | Completed / in progress / abandoned counts | Treat in-progress as censored, not failed |
| Blocked progress share | Share of progress updates flagged `blocked` | Stages where work stalls |
| Support load | Tickets per customer per onboarding day, by category and severity | Technical ticket volume, high severity |
| Ticket resolution time | `created_at` to `resolved_at` | Slow resolution by category or severity |

**Technical tickets** are those needing engineering or solutions-engineering
time: `erp_integration`, `inventory_sync`, `fulfillment_rules`,
`shipping_carrier` and `platform_bug`. Non-technical tickets are `how_to`,
`user_access`, `billing` and `feature_request`.

## Analysis guidance

- Compare within segment. Enterprise onboarding is inherently longer and more
  support-intensive than SMB onboarding.
- Normalize ticket counts by time in onboarding (exposure), not just per
  customer.
- Account for right-censoring: recent cohorts have had less time to complete,
  stall or abandon.
- Report associations as associations. Causal claims need evidence beyond
  co-occurrence.

## Not represented in the current data

- Retailer vs 3PL customer type (no column yet).
- Contract value, revenue and renewals.
- Onboarding owner (specialist or solutions engineer) per account.
- Specific ERP or storefront platform per customer.
