from __future__ import annotations

import argparse


def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """Üçgen membership function."""
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    return 0.0


def fuzzify_temperature(temp: float) -> dict[str, float]:
    """
    Sıcaklık için 3 dilsel değişken:
    - soğuk:   0-15-20
    - ılık:   15-20-25
    - sıcak:  20-30-35
    """
    cold = triangular_mf(temp, 0, 15, 20)
    warm = triangular_mf(temp, 15, 20, 25)
    hot = triangular_mf(temp, 20, 30, 35)
    return {"soguk": cold, "ilik": warm, "sicak": hot}


def rule_base(temp_fuzzy: dict[str, float]) -> dict[str, float]:
    """
    Basit kural tabanı:
    - Eğer sıcaklık soğuk ise fan_hizi = düşük
    - Eğer sıcaklık ılık ise fan_hizi = orta
    - Eğer sıcaklık sıcak ise fan_hizi = yüksek
    Fan hızı için üyelik dereceleri max-aggregation ile birleştiriliyor.
    """
    fan_low = temp_fuzzy["soguk"]
    fan_med = temp_fuzzy["ilik"]
    fan_high = temp_fuzzy["sicak"]

    return {
        "dusuk": fan_low,
        "orta": fan_med,
        "yuksek": fan_high,
    }


def defuzzify_fan_speed(fan_fuzzy: dict[str, float]) -> float:
    """
    Basit centroid benzeri bir yöntem:
    - düşük  -> 20
    - orta   -> 50
    - yüksek -> 80
    """
    centers = {"dusuk": 20.0, "orta": 50.0, "yuksek": 80.0}

    num = 0.0
    den = 0.0
    for label, mu in fan_fuzzy.items():
        c = centers[label]
        num += mu * c
        den += mu

    if den == 0.0:
        # Hiç kural ateşlenmediyse nötr bir değer dön
        return 50.0

    return num / den


def fuzzy_controller(temp: float) -> tuple[dict[str, float], dict[str, float], float]:
    temp_fuzzy = fuzzify_temperature(temp)
    fan_fuzzy = rule_base(temp_fuzzy)
    fan_crisp = defuzzify_fan_speed(fan_fuzzy)
    return temp_fuzzy, fan_fuzzy, fan_crisp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp",
        type=float,
        default=22.0,
        help="Ortam sıcaklığı (°C)",
    )
    args = parser.parse_args()

    temp = args.temp
    print(f"Girdi sıcaklık: {temp:.1f} °C")

    temp_fuzzy, fan_fuzzy, fan_crisp = fuzzy_controller(temp)

    print("\n=== Fuzzification (sıcaklık) ===")
    for label, mu in temp_fuzzy.items():
        print(f"{label:6s}: {mu:.3f}")

    print("\n=== Kural tabanı çıktısı (fan hızı membership) ===")
    for label, mu in fan_fuzzy.items():
        print(f"{label:6s}: {mu:.3f}")

    print(f"\n=== Defuzzification sonucu ===")
    print(f"Önerilen fan hızı (0-100 ölçeğinde): {fan_crisp:.2f}")


if __name__ == "__main__":
    main()
