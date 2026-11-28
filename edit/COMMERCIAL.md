# Commercial license for `kestrom`

## Free use

`kestrom` is free under any of three licenses, with one additional permission ([LICENSE](LICENSE)). In short (the license texts
decide):

- PolyForm Small Business 1.0.0, with our additional permission: use for the benefit of a company with fewer than 100 people
  (employees and independent contractors), counted together with every organisation it controls, is controlled by or is under
  common control with. Revenue does not matter.
- PolyForm Noncommercial 1.0.0: any noncommercial purpose; personal use for research, experiment, testing, study or hobbies without
  any anticipated commercial application; and use by charities, educational institutions, public research organisations, public
  safety or health organisations, environmental protection organisations and government institutions.
- PolyForm Free Trial 1.0.0: use to evaluate whether `kestrom` suits a particular application, for less than 32 consecutive
  calendar days, also on behalf of a company of any size. It does not allow distributing copies. If you need longer, write to us.

## Paid use

Any other use, such as a company of 100 people or more using `kestrom` for its business, needs a commercial license from CAPH TECH
Inc.

| People in the company | Price per year |
|---|---:|
| 100 to 999 | 3,000 USD |
| 1,000 or more | 10,000 USD |

People are counted as in Free use above, on the day of payment; a change during the year does not change that year's price.
Prices exclude taxes. We invoice in JPY on request, at the rate of the day.

What you get when the price of the company's row is paid: for one year from that day, use of any version of `kestrom` for the
benefit of the company named on the invoice or receipt, and of the organisations counted with it, is use for a permitted purpose
under the PolyForm Small Business License 1.0.0, whatever their size. All other terms of that license apply. After the year, use
needs a renewal, unless one of the free licenses covers it.

What is not in it:

- Support with a promised response time. Questions go to the issues of the repository.
- A license for your customers. Whoever receives `kestrom` from you, alone or inside a product, needs a license of their own, free
  or paid. A license that covers your customers is set by a written agreement.
- A warranty. The software comes as is (the No Liability section of the license).

The commercial license is governed by the laws of Japan, and the Tokyo District Court has exclusive jurisdiction in the first
instance over any dispute about it.

Before you buy: `kestrom ifc`, `kestrom inspect mirror`/`disarm`, and `kestrom-verify` (`generate torch`, `compare events`,
`diff`) are tested on both Apple Silicon (MLX) and Linux/NVIDIA CUDA (`torch`). `kestrom-verify container` (GGUF and safetensors
structural and metadata checks) is byte-level, runtime-independent, and unit-tested on Linux CI. `kestrom audit` supports
standalone GGUF container inspection (recording `mirror` as `SKIPPED`, so `NO_FINDINGS` on a GGUF file means no backdoor scan
ran) as well as safetensors folders; its full `container` + `diff` + `mirror` pipeline is unit-tested on Linux CI with NumPy
`nf4` (not yet run end-to-end on CUDA). `kestrom bridge` supports `mlx` and `torch` (`torch` runtime run end-to-end on macOS MPS
only; Linux NumPy CI covers its NumPy components, not yet run end-to-end on Linux/CUDA); `kv`, `rules`, and `rows` are tested on
Apple Silicon (MLX) only (`docs/guarantees.md`). For pre-purchase technical scope questions, use
[GitHub Discussions](https://github.com/CAPHTECH/kestrom/discussions). Use the free trial to check it on your application:
payments are not refunded.

## How to buy

Write to Ryoichi Izumita, r.izumita@caph.jp, with the company's name and its row in the table. We send a link for payment by
card, or an invoice (bank transfer). The license starts on the day of payment, under the terms of this page on that day; the
invoice or receipt names the company and the year.

The checker, `kestrom-verify`, is Apache-2.0 and needs no license from us, also for companies: anyone may check an edit.

## Enterprise, Partner & OEM Agreements

The Paid use license above already covers use of `kestrom` for the benefit of the licensed company (including its own internal
R&D and its own production systems) on an as-is basis, without an SLA or custom contract terms. Separate written agreements with
CAPH TECH Inc. are available for:

- **Enterprise Support & Procurement**: optional agreement for companies that require an NDA, vendor security onboarding, custom
  contract or invoicing terms, architecture review, or priority technical Q&A with defined response targets (reference range:
  25,000–50,000 USD / year; optional scoped PoC).
- **SIer & Security Audit Partner License**: required when running `kestrom` or `kestrom audit` for the benefit of third-party
  clients (outside PolyForm Small Business's own-company grant) or delivering `kestrom` / acceptance gates in client engagements.
- **OEM / Platform SDK License**: required when embedding or distributing `kestrom` (`inspect mirror` / `disarm`, `ifc`,
  `audit`) inside commercial scanners, registries, or platforms offered to customers (reference range: 50,000–150,000 USD / year).

For technical scoping, use [GitHub Discussions](https://github.com/CAPHTECH/kestrom/discussions); for commercial inquiries, write
as in **How to buy** above.
