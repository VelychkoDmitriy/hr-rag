"""Generate the internal HR documents of a fictional company: PDF, DOCX and CSV, with deliberate traps.

    python scripts/make_company_docs.py

The company "Polaris Demo LLC" (ТОВ «Поляріс Демо») is fictional; every document says so.
No real people: employees appear only as codes (E-0001). Traps for the evals:
- Employee Handbook v1 (2024) vs v2 (2025) contradict each other: remote days, annual leave, equipment allowance;
- the sales department leave policy looks like the company one but differs (similar-document trap);
- PDFs repeat a header and a "Page N of M" footer; the business-trip table runs across two pages;
- one PDF is a scan without a text layer (must be flagged for OCR, not indexed).
"""
import csv
import io
from pathlib import Path

import pymupdf as fitz  # renders the scan
from docx import Document
from docx.shared import Pt
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "company"
FONT = Path("C:/Windows/Fonts/arial.ttf")          # any TTF with Cyrillic; the files are generated once and committed
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
NOTICE_EN = "Fictional company. Course project document, not legal advice."
NOTICE_UK = "Вигадана компанія. Документ навчального проєкту, не є юридичною консультацією."


# ---------------------------------------------------------------- PDF

class Doc(FPDF):
    def __init__(self, header_text: str):
        super().__init__(format="A4")
        self.header_text = header_text
        self.add_font("A", "", str(FONT))
        self.add_font("A", "B", str(FONT_BOLD))
        self.set_auto_page_break(True, margin=20)
        self.alias_nb_pages()

    def header(self):
        self.set_font("A", "", 8)
        self.cell(0, 6, self.header_text, align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-14)
        self.set_font("A", "", 8)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="C")


def pdf_from_blocks(path: Path, header_text: str, blocks: list) -> None:
    """blocks: ("h1"|"h2"|"p"|"li", text) or ("table", [header, *rows], widths)."""
    pdf = Doc(header_text)
    pdf.add_page()
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            pdf.set_font("A", "B", 16); pdf.multi_cell(0, 9, block[1]); pdf.ln(2)
        elif kind == "h2":
            pdf.ln(2); pdf.set_font("A", "B", 12); pdf.multi_cell(0, 7, block[1]); pdf.ln(1)
        elif kind == "p":
            pdf.set_font("A", "", 11); pdf.multi_cell(0, 6.5, block[1]); pdf.ln(2)
        elif kind == "li":
            pdf.set_font("A", "", 11); pdf.multi_cell(0, 6.5, "• " + block[1]); pdf.ln(0.5)
        elif kind == "spacer":
            pdf.ln(block[1])
        elif kind == "table":
            rows, widths = block[1], block[2]
            pdf.set_font("A", "", 9)
            with pdf.table(col_widths=widths, text_align="LEFT", repeat_headings=0) as table:
                for row in rows:
                    r = table.row()
                    for cell in row:
                        r.cell(cell)
            pdf.ln(2)
    pdf.output(str(path))


HANDBOOK_COMMON = [
    ("h2", "1. About this handbook"),
    ("p", "This handbook explains the main rules of working at Polaris Demo LLC: working time, leave, remote work, "
          "equipment and conduct. Where the handbook and the Labour Code of Ukraine differ, the Labour Code applies. "
          "Questions go to the HR team at hr@polaris-demo.example."),
    ("h2", "2. Working time"),
    ("p", "The standard working week is 40 hours, Monday to Friday. Core hours, when everyone must be reachable, "
          "are from 11:00 to 16:00 Kyiv time. Overtime is allowed only with a written request from the team lead "
          "and is paid at double the hourly rate, as required by Article 106 of the Labour Code."),
    ("h2", "3. Probation"),
    ("p", "New employees have a probation period of up to three months. During probation the employee may resign "
          "with three days' notice. Probation does not apply to employees under 18 and to graduates in their first job."),
]


def handbook(version: str) -> list:
    v1 = version == "v1"
    effective = "2024-01-15" if v1 else "2025-03-01"
    remote_days = "two days per week" if v1 else "three days per week"
    leave_days = "24 calendar days" if v1 else "26 calendar days (24 required by law plus 2 company days)"
    equipment = "a one-time allowance of 3,000 UAH" if v1 else "an allowance of 1,500 UAH per year"
    blocks = [
        ("h1", f"Employee Handbook {version.upper()}"),
        ("p", f"Effective {effective}." + ("" if v1 else " Replaces Employee Handbook V1 of 2024-01-15.")),
        ("p", NOTICE_EN),
        *HANDBOOK_COMMON,
        ("h2", "4. Annual leave"),
        ("p", f"Every employee is entitled to annual paid leave of {leave_days}. Leave is planned in the leave schedule, "
              "which each department submits to HR by 15 December for the next year. At least one part of the leave "
              "must last no less than 14 calendar days."),
        ("p", "Unused leave is not paid out while the employee keeps working, except in the cases allowed by the "
              "Law of Ukraine On Leave. On dismissal, all unused days are compensated."),
        ("h2", "5. Remote work"),
        ("p", f"After probation, employees may work remotely up to {remote_days}. Remote days are agreed with the team "
              "lead in the HR portal at least one working day in advance. Warehouse and office-administration roles "
              "work on site."),
        ("p", f"For a home workplace, the company pays {equipment} for a chair, monitor or headset, against receipts."),
        ("h2", "6. Sick leave"),
        ("p", "An employee who falls ill informs the team lead before 10:00 on the first day. The electronic sick-leave "
              "certificate is registered by the doctor; HR receives it automatically and no paper copy is needed."),
        ("h2", "7. Conduct and confidentiality"),
        ("p", "Company and client data are used only for work. Laptops must use disk encryption and the corporate VPN "
              "outside the office. A lost device is reported to IT within one hour."),
        ("h2", "8. Public holidays"),
        ("p", "Under martial law, public holidays are working days in Ukraine (Article 73 of the Labour Code and Law "
              "No. 2136-IX). The company does not add its own days off on those dates; the extra company leave days "
              "can be used instead."),
        ("h2", "9. Performance reviews"),
        ("p", "Each employee has a review with the team lead twice a year, in June and December. The review covers "
              "the goals of the half-year, feedback from colleagues and a development plan. Salary changes are "
              "discussed only at the December review."),
        ("h2", "10. Learning"),
        ("p", "Courses, conferences and certifications are paid from the personal learning budget described in the "
              "Learning and Development Policy. Study leave is granted according to the Law of Ukraine On Leave."),
        ("h2", "11. Business trips"),
        ("p", "Business trips are arranged by an order of the CEO. Daily allowances and hotel limits are set by the "
              "Regulation on Business Trips; expenses above the limit are reimbursed only with written approval."),
        ("h2", "12. Leaving the company"),
        ("p", "An employee who resigns gives two weeks' notice through the HR portal. On the last day the employee "
              "returns the laptop and access badge and receives the final payment, including compensation for all "
              "unused leave days."),
        ("h2", "13. Contacts"),
        ("p", "HR team: hr@polaris-demo.example. IT service desk: it@polaris-demo.example. Security officer: "
              "security@polaris-demo.example. Urgent questions outside working hours go to the security officer."),
    ]
    return blocks


TRIP_ROWS = [["Destination", "Daily allowance", "Hotel limit per night", "Approval"]] + [
    [city, allowance, hotel, approval] for city, allowance, hotel, approval in [
        ("Kyiv", "600 UAH", "2,500 UAH", "Team lead"), ("Lviv", "600 UAH", "2,200 UAH", "Team lead"),
        ("Odesa", "600 UAH", "2,200 UAH", "Team lead"), ("Dnipro", "600 UAH", "2,000 UAH", "Team lead"),
        ("Kharkiv", "600 UAH", "2,000 UAH", "Team lead and security officer"),
        ("Other cities of Ukraine", "500 UAH", "1,800 UAH", "Team lead"),
        ("Poland", "50 EUR", "110 EUR", "Head of department"), ("Germany", "60 EUR", "140 EUR", "Head of department"),
        ("Other EU countries", "55 EUR", "130 EUR", "Head of department"), ("United Kingdom", "55 GBP", "150 GBP", "CEO"),
        ("USA", "70 USD", "180 USD", "CEO"), ("Other countries", "50 USD", "120 USD", "CEO"),
    ]]


def business_trips() -> list:
    return [
        ("h1", "Положення про службові відрядження"),
        ("p", "Діє з 2025-01-10. " + NOTICE_UK),
        ("h2", "1. Загальні положення"),
        ("p", "Відрядження — поїздка працівника за розпорядженням керівника для виконання службового доручення поза "
              "місцем постійної роботи. Відрядження оформлюється наказом, у якому зазначають мету, місце і строк. "
              "За працівником на весь час відрядження зберігається середній заробіток."),
        ("h2", "2. Добові та ліміти на проживання"),
        ("p", "Добові виплачуються за кожен день відрядження, включно з днями виїзду і повернення. Ліміти наведено "
              "в таблиці; витрати понад ліміт компенсуються лише з письмового дозволу особи, що погоджує."),
        ("spacer", 150),                     # pushes the table across the page break: a split-table trap
        ("table", TRIP_ROWS, (34, 22, 26, 38)),
        ("h2", "3. Звіт про відрядження"),
        ("p", "Протягом п'яти робочих днів після повернення працівник подає звіт про використання коштів разом із "
              "квитками, рахунками готелю та чеками. Невикористаний аванс повертається в той самий строк."),
        ("h2", "4. Відрядження під час воєнного стану"),
        ("p", "Поїздки до областей, де тривають активні бойові дії, не плануються. Для поїздок до Харкова потрібне "
              "додаткове погодження з офіцером безпеки компанії."),
    ]


def bonus_policy() -> list:
    return [
        ("h1", "Положення про преміювання працівників"),
        ("p", "Діє з 2025-02-01. " + NOTICE_UK),
        ("h2", "1. Види премій"),
        ("li", "Квартальна премія — до 20% окладу за підсумками виконання цілей кварталу (OKR)."),
        ("li", "Річна премія — до одного окладу, якщо компанія виконала річний план виручки."),
        ("li", "Разова премія — за завершення ключового проєкту, за поданням керівника відділу."),
        ("h2", "2. Хто має право"),
        ("p", "Квартальна премія нараховується працівникам, які пропрацювали в кварталі не менше двох місяців і не "
              "мають дисциплінарних стягнень. Працівники на випробувальному строку квартальну премію не отримують."),
        ("h2", "3. Порядок нарахування"),
        ("p", "Керівник відділу подає оцінку виконання цілей до 10 числа місяця, що настає за кварталом. Премія "
              "виплачується разом із заробітною платою за перший місяць наступного кварталу."),
        ("h2", "4. Позбавлення премії"),
        ("p", "Працівника може бути позбавлено премії повністю або частково за порушення трудової дисципліни, "
              "що оформлено наказом. Позбавлення премії не є дисциплінарним стягненням."),
    ]


def make_scan(path: Path) -> None:
    """A signed acknowledgement as an image-only PDF: no text layer at all."""
    tmp = io.BytesIO()
    pdf = Doc("Polaris Demo LLC — scanned document")
    pdf.add_page()
    pdf.set_font("A", "B", 14); pdf.multi_cell(0, 8, "Acknowledgement of the Information Security Policy"); pdf.ln(2)
    pdf.set_font("A", "", 11)
    pdf.multi_cell(0, 6, "I confirm that I have read the Information Security Policy of Polaris Demo LLC and will "
                         "follow it. Employee code: E-0042. Date: 2025-04-02. Signature: ______. " + NOTICE_EN)
    pdf.output(tmp)
    src = fitz.open(stream=tmp.getvalue(), filetype="pdf")
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=110, colorspace=fitz.csGRAY)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
    out.save(str(path))


# ---------------------------------------------------------------- DOCX

def docx_from_blocks(path: Path, blocks: list) -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10.5)
    for block in blocks:
        kind = block[0]
        if kind == "title":
            doc.add_paragraph(block[1], style="Title")
        elif kind == "h1":
            doc.add_paragraph(block[1], style="Heading 1")
        elif kind == "p":
            doc.add_paragraph(block[1])
        elif kind == "num":
            doc.add_paragraph(block[1], style="List Number")
        elif kind == "li":
            doc.add_paragraph(block[1], style="List Bullet")
        elif kind == "table":
            rows = block[1]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    table.cell(i, j).text = value
    doc.save(str(path))


def leave_policy(sales: bool) -> list:
    title = "Політика щорічних відпусток відділу продажів" if sales else "Політика щорічних відпусток"
    extra = ("Працівникам відділу продажів додаткові дні за стаж не надаються; натомість у грудні діє заборона на "
             "відпустки понад 5 робочих днів через закриття річного плану.") if sales else (
             "За кожні повні три роки роботи в компанії працівник отримує 1 додатковий день відпустки, але не більше "
             "3 днів сукупно.")
    return [
        ("title", title),
        ("p", "Діє з 2025-03-01. " + NOTICE_UK),
        ("h1", "1. Тривалість"),
        ("p", "Щорічна основна відпустка становить 26 календарних днів: 24 дні за законом і 2 дні від компанії. "
              "Працівникам грейдів G3 і G4 — 28 календарних днів (див. штатний розпис). "
              "Святкові та неробочі дні до тривалості відпустки не включаються."),
        ("p", extra),
        ("h1", "2. Планування"),
        ("num", "До 1 грудня працівник погоджує з керівником бажані дати на наступний рік."),
        ("num", "До 15 грудня керівник подає графік відпусток відділу до HR."),
        ("num", "Зміни до графіка погоджуються з керівником не пізніше ніж за 14 днів до початку відпустки."),
        ("h1", "3. Поділ і перенесення"),
        ("p", "Відпустку можна поділити на частини, але одна з них має бути не менше 14 календарних днів. "
              "Невикористану відпустку можна перенести на наступний рік за письмовою заявою працівника."),
    ]


def remote_policy() -> list:
    return [
        ("title", "Політика дистанційної роботи"),
        ("p", "Діє з 2025-03-01, узгоджена з Employee Handbook V2. " + NOTICE_UK),
        ("h1", "1. Хто може працювати дистанційно"),
        ("p", "Працівники, які пройшли випробувальний строк, можуть працювати дистанційно до трьох днів на тиждень. "
              "Повністю дистанційна робота оформлюється окремим трудовим договором про дистанційну роботу "
              "відповідно до статті 60-2 Кодексу законів про працю."),
        ("h1", "2. Обладнання"),
        ("table", [["Що", "Хто надає", "Ліміт"], ["Ноутбук", "Компанія", "—"], ["Гарнітура", "Компанія", "—"],
                   ["Монітор, крісло", "Працівник, компенсація за чеками", "1 500 грн на рік"],
                   ["Інтернет", "Працівник", "не компенсується"]]),
        ("h1", "3. Безпека"),
        ("li", "Робота з даними клієнтів лише через корпоративний VPN."),
        ("li", "Заборонено використовувати публічний Wi-Fi без VPN."),
        ("li", "Про втрату пристрою повідомити IT протягом однієї години."),
    ]


def onboarding() -> list:
    return [
        ("title", "Онбординг нового працівника"),
        ("p", "Діє з 2024-09-01. " + NOTICE_UK),
        ("h1", "До першого дня"),
        ("num", "HR надсилає лист із датою виходу, адресою офісу та переліком документів."),
        ("num", "Працівник надає паспорт, РНОКПП та диплом через захищений HR-портал."),
        ("num", "IT готує ноутбук і обліковий запис."),
        ("h1", "Перший тиждень"),
        ("num", "Інструктаж з охорони праці та пожежної безпеки — у перший день, до початку роботи."),
        ("num", "Знайомство з наставником (buddy), який супроводжує працівника перші три місяці."),
        ("num", "Постановка цілей на випробувальний строк разом із керівником."),
        ("h1", "Кінець випробувального строку"),
        ("p", "За два тижні до завершення випробувального строку керівник проводить зустріч-оцінку. Якщо працівник "
              "не пройшов випробування, роботодавець повідомляє його письмово за три дні."),
    ]


def dismissal() -> list:
    return [
        ("title", "Порядок звільнення"),
        ("p", "Діє з 2024-09-01. " + NOTICE_UK),
        ("h1", "1. Звільнення за власним бажанням"),
        ("p", "Працівник подає заяву через HR-портал за два тижні до дати звільнення (стаття 38 КЗпП). Строк можна "
              "скоротити за згодою сторін."),
        ("h1", "2. Останній робочий день"),
        ("num", "Повернути ноутбук, перепустку та інше обладнання до кінця дня."),
        ("num", "Пройти вихідне інтерв'ю з HR (необов'язково)."),
        ("num", "Отримати остаточний розрахунок, включно з компенсацією за невикористану відпустку."),
        ("h1", "3. Розрахунок"),
        ("p", "Усі суми, належні працівникові, виплачуються в день звільнення, як вимагає стаття 116 КЗпП."),
    ]


def training() -> list:
    return [
        ("title", "Політика навчання і розвитку"),
        ("p", "Діє з 2025-01-10. " + NOTICE_UK),
        ("h1", "Бюджет"),
        ("p", "Кожен працівник має річний бюджет на навчання 15 000 грн: курси, конференції, сертифікації, книги. "
              "Невикористаний бюджет на наступний рік не переноситься."),
        ("h1", "Погодження"),
        ("p", "Навчання до 5 000 грн погоджує керівник команди, понад 5 000 грн — керівник відділу і HR. Якщо "
              "працівник звільняється протягом 6 місяців після сертифікації вартістю понад 10 000 грн, він "
              "компенсує її вартість пропорційно."),
        ("h1", "Навчальні відпустки"),
        ("p", "Відпустки у зв'язку з навчанням надаються відповідно до статті 15 Закону України «Про відпустки»."),
    ]


def overtime() -> list:
    return [
        ("title", "Понаднормова робота і робота у вихідні"),
        ("p", "Діє з 2024-09-01. " + NOTICE_UK),
        ("h1", "Погодження"),
        ("p", "Понаднормова робота допускається лише з письмового запиту керівника і не може перевищувати 4 годин "
              "протягом двох днів підряд і 120 годин на рік (стаття 65 КЗпП)."),
        ("h1", "Оплата"),
        ("p", "Понаднормові години оплачуються в подвійному розмірі годинної ставки (стаття 106 КЗпП). Робота у "
              "вихідний день компенсується за згодою сторін іншим днем відпочинку або оплатою в подвійному розмірі."),
    ]


def sick_leave() -> list:
    return [
        ("title", "Лікарняні"),
        ("p", "Діє з 2024-09-01. " + NOTICE_UK),
        ("h1", "Що робити"),
        ("num", "Повідомити керівника до 10:00 першого дня хвороби."),
        ("num", "Звернутися до сімейного лікаря, який відкриває електронний листок непрацездатності."),
        ("num", "Після закриття листка повідомити HR про вихід на роботу."),
        ("h1", "Оплата"),
        ("p", "Перші п'ять днів непрацездатності оплачує компанія, починаючи з шостого — Пенсійний фонд України "
              "як страховик. Компанія доплачує до 100% середньої зарплати за перші 10 днів хвороби на рік."),
    ]


# ---------------------------------------------------------------- CSV

STAFF = [
    ("ENG-01", "Розробка", "Junior Developer", "G1", 30000, 45000, 26),
    ("ENG-02", "Розробка", "Middle Developer", "G2", 55000, 80000, 26),
    ("ENG-03", "Розробка", "Senior Developer", "G3", 90000, 130000, 28),
    ("ENG-04", "Розробка", "Tech Lead", "G4", 130000, 170000, 28),
    ("QA-01", "Тестування", "QA Engineer", "G2", 40000, 65000, 26),
    ("SAL-01", "Продажі", "Sales Manager", "G2", 35000, 50000, 26),
    ("SAL-02", "Продажі", "Head of Sales", "G4", 90000, 120000, 28),
    ("HR-01", "HR", "HR Generalist", "G2", 35000, 50000, 26),
    ("HR-02", "HR", "Recruiter", "G2", 32000, 48000, 26),
    ("FIN-01", "Фінанси", "Accountant", "G2", 35000, 50000, 26),
    ("ADM-01", "Адміністрація", "Office Manager", "G1", 25000, 32000, 26),
    ("WH-01", "Склад", "Warehouse Operator", "G1", 22000, 28000, 26),
]


def write_csv(path: Path, header: list, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def leave_schedule() -> list:
    months = ["2026-01", "2026-03", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-12"]
    rows = []
    for i in range(1, 25):
        code = STAFF[(i * 5) % len(STAFF)][0]
        rows.append([f"E-{i:04d}", code, months[i % len(months)], 14 if i % 3 else 7, "погоджено" if i % 4 else "на погодженні"])
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = "Polaris Demo LLC — Internal"
    pdf_from_blocks(OUT / "employee_handbook_v1.pdf", header, handbook("v1"))
    pdf_from_blocks(OUT / "employee_handbook_v2.pdf", header, handbook("v2"))
    pdf_from_blocks(OUT / "polozhennia_vidriadzhennia.pdf", "ТОВ «Поляріс Демо» — для внутрішнього користування", business_trips())
    pdf_from_blocks(OUT / "polozhennia_premiiuvannia.pdf", "ТОВ «Поляріс Демо» — для внутрішнього користування", bonus_policy())
    make_scan(OUT / "infosec_acknowledgement_scan.pdf")
    for name, blocks in [("polityka_vidpustok", leave_policy(False)), ("polityka_vidpustok_prodazhi", leave_policy(True)),
                         ("polityka_dystantsiinoi_roboty", remote_policy()), ("onboarding", onboarding()),
                         ("poriadok_zvilnennia", dismissal()), ("polityka_navchannia", training()),
                         ("ponadnormova_robota", overtime()), ("likarniani", sick_leave())]:
        docx_from_blocks(OUT / f"{name}.docx", blocks)
    write_csv(OUT / "shtatnyi_rozpys.csv",
              ["код_посади", "відділ", "посада", "грейд", "оклад_мін_грн", "оклад_макс_грн", "щорічна_відпустка_днів"], STAFF)
    write_csv(OUT / "hrafik_vidpustok_2026.csv",
              ["код_працівника", "код_посади", "місяць", "днів", "статус"], leave_schedule())
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<38} {p.stat().st_size:>8,} B")


if __name__ == "__main__":
    main()
