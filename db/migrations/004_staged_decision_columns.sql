-- PlainCents ML-G: Preview must show the decision Confirm will store.
--
-- Preview previously staged only `predicted_category` (the raw model output)
-- and Confirm independently re-derived structural-ambiguity routing and
-- remembered corrections. The two therefore disagreed exactly on the rows
-- where the difference mattered.
--
-- These columns let the ONE decision path
-- (backend/services/category_decision.py) run once at Preview and persist
-- its complete result, so Confirm re-validates rather than re-decides:
--
--   merchant_key            stable, bank-scoped merchant identity used for
--                           correction-memory lookup (NULL when the text
--                           names nothing)
--   remembered_category     a prior GENUINE user correction that will be
--                           written to transactions.confirmed_category on
--                           confirm. NULL means no correction is remembered.
--                           This is never a model output.
--   decision_source         'model' | 'structural_other' |
--                           'low_confidence_other' -- why predicted_category
--                           is what it is, shown in the Preview table.
--   model_category          what the classifier alone said, kept even when
--                           the abstention policy overrode it, so a preview
--                           row stays auditable.
ALTER TABLE staged_transactions ADD COLUMN merchant_key TEXT;
ALTER TABLE staged_transactions ADD COLUMN remembered_category TEXT;
ALTER TABLE staged_transactions ADD COLUMN decision_source TEXT;
ALTER TABLE staged_transactions ADD COLUMN model_category TEXT;
