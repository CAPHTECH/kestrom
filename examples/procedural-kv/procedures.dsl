# 4-head standalone DSL for procedural memory KV entries
in cond_logic: match "\\bKestrel-Gate\\b"
h1 u_cond_logic: any(cond_logic)
h2 v_cond_logic: any(u_cond_logic)
out cond_logic/hit: 1*v_cond_logic

in tiered_fee: match "\\bHarbor-Pass\\b"
h1 u_tiered_fee: any(tiered_fee)
h2 v_tiered_fee: any(u_tiered_fee)
out tiered_fee/hit: 1*v_tiered_fee

in cipher_rule: match "\\bSpire-Tag\\b"
h1 u_cipher_rule: any(cipher_rule)
h2 v_cipher_rule: any(u_cipher_rule)
out cipher_rule/hit: 1*v_cipher_rule

in triage_policy: match "\\bAtlas-Ops\\b"
h1 u_triage_policy: any(triage_policy)
h2 v_triage_policy: any(u_triage_policy)
out triage_policy/hit: 1*v_triage_policy
