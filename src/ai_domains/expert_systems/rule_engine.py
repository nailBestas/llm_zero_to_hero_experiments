from __future__ import annotations

import argparse


# Çok basit bir semptom -> öneri uzman sistemi.
# Gerçek tıbbi sistem DEĞİL, sadece kural tabanlı mantığı göstermek için.
RULES = [
    {
        "conditions": {"ates": "evet", "oksuruk": "evet"},
        "advice": "Yüksek ateş ve öksürük var; bir doktora görünmen önerilir.",
    },
    {
        "conditions": {"ates": "evet", "bas_agrisi": "evet"},
        "advice": "Ateş ve baş ağrısı belirtileri var; dinlen ve şikayetler devam ederse doktora git.",
    },
    {
        "conditions": {"ates": "hayir", "oksuruk": "evet"},
        "advice": "Sadece öksürük var; soğuk algınlığı olabilir, bol su iç ve durumu gözle.",
    },
    {
        "conditions": {"ates": "hayir", "oksuruk": "hayir", "yorgunluk": "evet"},
        "advice": "Yorgunluk hissediyorsun; uyku ve stres durumunu düzenlemeyi dene.",
    },
]


def ask_yes_no(prompt: str) -> str:
    while True:
        val = input(prompt + " (evet/hayir): ").strip().lower()
        if val in {"evet", "hayir"}:
            return val
        print("Lütfen sadece 'evet' veya 'hayir' yaz.")


def collect_facts() -> dict[str, str]:
    print("=== Basit Semptom Uzman Sistemi ===")
    print("Not: Bu sadece eğitim amaçlı bir örnek, tıbbi tavsiye DEĞİLDİR.\n")

    facts = {}
    facts["ates"] = ask_yes_no("Ateşin var mı?")
    facts["oksuruk"] = ask_yes_no("Öksürüğün var mı?")
    facts["bas_agrisi"] = ask_yes_no("Baş ağrın var mı?")
    facts["yorgunluk"] = ask_yes_no("Kendini yorgun hissediyor musun?")
    return facts


def match_rule(facts: dict[str, str], rule: dict) -> bool:
    for key, expected in rule["conditions"].items():
        if facts.get(key) != expected:
            return False
    return True


def infer(facts: dict[str, str]) -> list[str]:
    advices = []
    for rule in RULES:
        if match_rule(facts, rule):
            advices.append(rule["advice"])
    return advices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Komut satırı üzerinden sorular sorarak çalıştır.",
    )
    args = parser.parse_args()

    if args.interactive:
        facts = collect_facts()
    else:
        # Demo amaçlı sabit bir örnek
        facts = {
            "ates": "evet",
            "oksuruk": "evet",
            "bas_agrisi": "hayir",
            "yorgunluk": "evet",
        }
        print("Demo facts:", facts)

    advices = infer(facts)

    print("\n=== Uzman Sistem Önerileri ===")
    if not advices:
        print("Belirtilere uyan özel bir kural bulunamadı. Durumu gözlemlemeye devam et.")
    else:
        for i, adv in enumerate(advices, start=1):
            print(f"{i}. {adv}")


if __name__ == "__main__":
    main()
