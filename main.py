# Peramal penonton bersamaan (CCU) Twitch pada hari peluncuran sebuah game besar.
#
# Cara kerja, singkatnya: untuk tiap jam ramalan, kami mempelajari dari
# peluncuran-peluncuran besar sebelumnya bagaimana penonton pada jam itu
# biasanya bergerak relatif terhadap jam pertama. Pergerakan itu kami
# ukur dalam skala logaritma: absisnya log penonton jam pertama, ordinatnya
# log rasio penonton jam itu terhadap penonton jam pertama. Sebuah garis
# lurus kuadrat terkecil menghubungkan keduanya, lalu titik tengah ramalan
# ditarik sedikit (seperempat jalan) ke arah median rasio-rasio yang
# teramati, supaya satu peluncuran yang menyimpang tidak menarik ramalan
# terlalu jauh. Sebaran ramalan diambil dari sebaran sisa-sisa garis itu
# (selisih antarkuartil dibagi 1,349, kira-kira simpangan baku bila sisanya
# normal), dijepit agar tidak terlalu sempit atau terlalu lebar, dan
# dilebarkan seperempat bila jam pertama kasus ini berada di luar jangkauan
# jam pertama yang pernah diamati. Lima kuantil (5%, 25%, 50%, 75%, 95%)
# lalu dibangun dari titik tengah dan sebaran itu dengan titik-titik
# normal baku.
#
# Mengapa skala lebar kipas 0,73? Titik tengah ramalan tidak bergantung
# pada skala ini, hanya lebar selang antar kuantilnya. Pada empat kasus
# practice, cakupan 90 persen terukur 0,8696 pada skala 0,60 (80 dari 92
# jam tercakup) dan naik menjadi 0,8913 pada skala 0,73 (82 dari 92 jam),
# sementara kerugian practice hanya naik sedikit sebagai harga mekanis dari
# selang yang lebih lebar. Skala 0,73 menempatkan cakupan di tengah pita
# target 0,88-0,92 tanpa mengejar jam-jam yang meleset jauh dari titik
# tengah, yang akan menaikkan kerugian jauh lebih besar.
#
# Hanya pustaka standar, tanpa jaringan, deterministik: masukan yang sama
# selalu menghasilkan keluaran yang sama.

import json
import math
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Pengali lebar kipas kuantil. Lihat komentar kepala berkas untuk alasannya.
SPREAD_SCALE = 0.73

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
FORECAST_HOURS = list(range(2, 25))


def parse_time(text):
    """Ubah stempel waktu ISO 8601 (dengan akhiran Z) menjadi datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def quantile(sorted_values, p):
    """Kuantil dengan interpolasi linear pada nilai-nilai yang sudah terurut."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def median(values):
    return quantile(sorted(values), 0.5)


def hourly_viewers(comp):
    """Peta penonton per jam pada hari rilis sebuah comparable.

    Tiap titik dengan penonton positif dipetakan ke nomor jam dari selisih
    terhadap day_start; hanya jam 1 sampai 24 yang disimpan.
    """
    day_start = parse_time(comp["day_start"])
    per_hour = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers", 0)
        if viewers is None or viewers <= 0:
            continue
        delta_seconds = (parse_time(point["from"]) - day_start).total_seconds()
        hour = int(delta_seconds // 3600) + 1
        if 1 <= hour <= 24:
            per_hour[hour] = viewers
    return per_hour


def training_launches(case):
    """Peluncuran latih: hari rilis memuat jam pertama dan >= 20 dari 24 jam.

    Dipakai himpunan 'bersih' (unlock memuat penanda Tier: comparable)
    bila jumlahnya sedikitnya 8; bila tidak, semua peluncuran yang dapat
    dipakai.
    """
    usable = []
    for comp in case.get("comparables", []):
        per_hour = hourly_viewers(comp)
        if 1 in per_hour and len(per_hour) >= 20:
            usable.append((comp, per_hour))
    clean = [
        (comp, per_hour)
        for comp, per_hour in usable
        if "Tier: comparable" in (comp.get("unlock") or "")
    ]
    chosen = clean if len(clean) >= 8 else usable
    return [per_hour for _, per_hour in chosen]


def forecast_hour(hour, hour1, train):
    """Lima kuantil ramalan untuk satu jam, dari peluncuran-peluncuran latih."""
    log_hour1 = math.log(hour1)
    pairs = []
    for per_hour in train:
        if hour in per_hour:
            first = per_hour[1]
            pairs.append((math.log(first), math.log(per_hour[hour] / first)))

    if len(pairs) < 4:
        centre = 0.0
        spread = 0.15 * math.sqrt(hour - 1)
    else:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / sxx
        else:
            slope = 0.0
        intercept = mean_y - slope * mean_x

        centre = intercept + slope * log_hour1
        # Tarik seperempat jalan menuju median rasio-rasio log yang teramati.
        centre = centre + 0.25 * (median(ys) - centre)
        # Jepit ke dalam rentang rasio log yang teramati.
        centre = min(max(centre, min(ys)), max(ys))

        residuals = sorted(y - (intercept + slope * x) for x, y in pairs)
        iqr = quantile(residuals, 0.75) - quantile(residuals, 0.25)
        spread = iqr / 1.349
        spread = min(max(spread, 0.05), 0.5)
        if log_hour1 < min(xs) or log_hour1 > max(xs):
            spread = min(spread * 1.25, 0.5)

    quantiles = []
    for z in Z_SCORES:
        value = hour1 * math.exp(centre + z * spread * SPREAD_SCALE)
        value = min(max(value, 1.0), 1e9)
        # Bulatkan ke dua desimal, setengah ke atas.
        value = float(
            Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
        quantiles.append(value)
    # Paksa tidak menurun: naikkan nilai apa pun yang di bawah kuantil sebelumnya.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    # Bilangan bulat hasil pembulatan dikeluarkan sebagai bilangan bulat.
    return [int(v) if v == int(v) else v for v in quantiles]


def forecast_case(case):
    """Seluruh baris ramalan (jam 2..24) untuk satu kasus."""
    hour1 = 1
    for row in case.get("known", []):
        if row.get("hour") == 1:
            hour1 = row.get("viewers", 1)
            break
    hour1 = max(1, hour1 if hour1 else 1)

    train = training_launches(case)
    rows = []
    for hour in FORECAST_HOURS:
        rows.append({"hour": hour, "q": forecast_hour(hour, hour1, train)})
    return rows


def validate_rows(rows):
    """Periksa setiap baris ramalan sebelum ditulis.

    Pemeriksaan ini tidak mengubah keluaran apa pun: ia hanya menolak
    menulis bila ada baris yang melanggar bentuk yang dijanjikan
    (tepat lima kuantil, jam 2 sampai 24 masing-masing tepat sekali,
    semua kuantil berhingga, positif, dan tidak menurun).
    """
    if len(rows) != 23:
        raise ValueError("expected 23 forecast rows, got %d" % len(rows))
    seen = set()
    for row in rows:
        hour = row.get("hour")
        if hour in seen:
            raise ValueError("duplicate forecast hour %r" % (hour,))
        seen.add(hour)
        quantiles = row.get("q")
        if not isinstance(quantiles, list) or len(quantiles) != 5:
            raise ValueError(
                "hour %r: expected exactly 5 quantiles, got %r" % (hour, quantiles)
            )
        for value in quantiles:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("hour %r: non-finite quantile %r" % (hour, value))
            if value <= 0:
                raise ValueError("hour %r: non-positive quantile %r" % (hour, value))
        for prev, cur in zip(quantiles, quantiles[1:]):
            if cur < prev:
                raise ValueError(
                    "hour %r: quantiles decrease (%r > %r)" % (hour, prev, cur)
                )
    if seen != set(FORECAST_HOURS):
        raise ValueError("forecast hours must be exactly 2..24, got %s" % sorted(seen))


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    in_paths = sorted(input_dir.glob("*.in"))
    if not in_paths:
        raise SystemExit("no .in case files found in %s" % input_dir)
    for in_path in in_paths:
        case = json.loads(in_path.read_text())
        rows = forecast_case(case)
        validate_rows(rows)
        payload = {"hours": rows}
        out_path = output_dir / (in_path.stem + ".out")
        out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        if not out_path.is_file():
            raise SystemExit("failed to write %s" % out_path)


if __name__ == "__main__":
    main()
