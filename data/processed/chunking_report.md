# Chunking: structure-aware vs fixed windows

dataset `5e03ca05499fe1bf`, 36 documents. Fixed: 400 tokens, overlap 60. Structure: section > article > whole paragraphs up to 450 tokens.

| set | strategy | chunks | avg tok | max tok | start mid-sentence | cut table rows | two sections in one chunk | knows its section |
|---|---|---|---|---|---|---|---|---|
| all | structure | 3661 | 305 | 675 | 6.2% | 0.0% | 0.6% | 99.3% |
| all | fixed | 3029 | 398 | 401 | 89.7% | 0.1% | 49.9% | 0.0% |
| company_data | structure | 3 | 373 | 447 | 0.0% | 0.0% | 0.0% | 100.0% |
| company_data | fixed | 3 | 372 | 400 | 33.3% | 66.7% | 0.0% | 0.0% |
| company_policy | structure | 57 | 114 | 337 | 0.0% | 0.0% | 19.3% | 98.2% |
| company_policy | fixed | 23 | 311 | 400 | 30.4% | 0.0% | 95.7% | 0.0% |
| guidance | structure | 758 | 390 | 675 | 8.0% | 0.0% | 0.0% | 100.0% |
| guidance | fixed | 770 | 398 | 400 | 89.9% | 0.0% | 42.5% | 0.0% |
| ilo | structure | 216 | 267 | 448 | 8.3% | 0.0% | 0.5% | 93.5% |
| ilo | fixed | 168 | 399 | 401 | 86.3% | 0.0% | 66.7% | 0.0% |
| law | structure | 2627 | 288 | 563 | 5.6% | 0.0% | 0.3% | 99.7% |
| law | fixed | 2065 | 399 | 401 | 90.7% | 0.0% | 50.8% | 0.0% |

## Near-duplicate documents (kept: versions are marked with is_current)

- `employee_handbook_v1` ~ `employee_handbook_v2` (Jaccard 0.891)
- `employee_handbook_v2` ~ `employee_handbook_v1` (Jaccard 0.891)

## Example: the business-trip table (it runs across two PDF pages)

Fixed window (the chunk that holds the table):

```
# Положення про службові відрядження

Діє з 2025-01-10. Вигадана компанія. Документ навчального проєкту, не є юридичною консультацією.

## 1. Загальні положення

Відрядження — поїздка працівника за розпорядженням керівника для виконання службового доручення поза місцем постійної роботи. Відрядження оформлюється наказом, у якому зазначають мету, місце і строк. За працівником на весь час відрядження зберігається середній заробіток.

## 2. Добові та ліміти на проживання

Добові виплачуються за кожен день відрядження, включно з днями виїзду і повернення. Ліміти наведено в таблиці; витрати понад ліміт компенсуються лише з письмового дозволу особи, що погоджує.

|Destination|Daily allowance|Hotel limit per night|Approval|
|---|---|---|---|
|Kyiv|
```

Structure-aware (the table chunk):

```
## 2. Добові та ліміти на проживання

Добові виплачуються за кожен день відрядження, включно з днями виїзду і повернення. Ліміти наведено в таблиці; витрати понад ліміт компенсуються лише з письмового дозволу особи, що погоджує.

|Destination|Daily allowance|Hotel limit per night|Approval|
|---|---|---|---|
|Kyiv|600 UAH|2,500 UAH|Team lead|
|Lviv|600 UAH|2,200 UAH|Team lead|
|Odesa|600 UAH|2,200 UAH|Team lead|
|Dnipro|600 UAH|2,000 UAH|Team lead|
|Kharkiv|600 UAH|2,000 UAH|Team lead and security officer|
|Other cities of Ukraine|500 UAH|1,800 UAH|Team lead|
|Poland|50 EUR|110 EUR|Head of department|
|Germany|60 EUR|140 EUR|Head of department|
|Other EU countries|55 EUR|130 EUR|Head of department|
|United Kingdom|55 GBP|150 GBP|CEO|
|USA|70 USD|180 USD|CEO|
|Other countries|50 USD|120 USD|CEO|
```
