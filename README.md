# MRTO — Methanol Reformer Tube & Operation Optimizer

سامانه هوشمند پایش سلامت لوله‌های ریفورمر و بهینه‌سازی نسبت بخار به کربن
**پتروشیمی مرجان — سطح آمادگی فناوری TRL 5**

---

## 📌 معرفی محصول

سامانه **MRTO** یک راهکار هوشمند مبتنی بر یادگیری ماشین برای واحد تولید متانول است که چهار مسئله کلیدی کوره ریفورمر را به صورت یکپارچه حل می‌کند:

1. **ماژول ۱ — پایش توزیع دما و شناسایی نقاط داغ لوله‌ها**
2. **ماژول ۲ — برآورد عمر باقیمانده (RUL) و پیش‌بینی خرابی لوله‌ها**
3. **ماژول ۳ — بهینه‌سازی نسبت بخار به کربن (S/C) و شرایط بهره‌برداری**
4. **ماژول ۴ — پایش عملکرد مشعل‌ها، کاتالیست و افت فشار**

این سند، خلاصه‌ای از معماری، نحوه اجرا، خط لوله CI/CD و نتایج آموزش را ارائه می‌دهد. مستندات کامل کسب‌واره (BRS) و نرم‌افزاری (SRS) در پوشه `docs/` قرار دارند.

---

## 🏗 معماری کلی

```
┌──────────────────────────────────────────────────────────────────┐
│  داشبورد وب واکنش‌گرا (frontend/index.html — تک‌فایل HTML/CSS/JS) │
└────────────────────────────┬─────────────────────────────────────┘
                             │ REST/JSON
┌────────────────────────────▼─────────────────────────────────────┐
│  nginx Reverse Proxy (deploy/nginx.conf)                         │
│  ├── rate-limit + security headers                                │
│  └── /api/* → backend:8000   |   /* → static dashboard             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  بک‌اند FastAPI (backend/mrto/app.py)                            │
│  ├── /api/v1/health, /auth, /tubes, /temperature-map            │
│  ├── /rul, /optimization, /burners, /alerts, /reports, /metrics │
│  ├── احراز هویت JWT، RBAC، لاگ ممیزی                           │
└──────┬───────────────────────────────────────┬──────────────────┘
       │                                         │
┌──────▼────────────────┐              ┌────────▼──────────────────┐
│  SQLite / PVC          │              │  مدل‌های ML (joblib)     │
│  (backend/mrto/db.py)  │              │  module1/2/3/4           │
└────────────────────────┘              └───────────────────────────┘
       ▲                                           ▲
       │                                           │
┌──────┴───────────────────────────────────────────┴──────────────┐
│  خط لوله آموزش (mrto/train_models.py)                           │
│  ├── داده مصنوعی فیزیک‌محور (synthetic_data.py)                  │
│  ├── داده واقعی از UCI (real_data.py)                           │
│  └── آموزش چهار مدل + گزارش معیارها → training_report.json      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 شروع سریع

### پیش‌نیازها
- Python 3.12+, Docker 24+, docker compose v2

### ۱) اجرای محلی (توسعه)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# تولید داده + آموزش مدل‌ها
python -m mrto.synthetic_data ../data/raw
python -m mrto.train_models --out-dir ../data --models-dir ../models \
  --report ../training_report.json

# اجرای بک‌اند
uvicorn mrto.app:app --reload --port 8000

# اجرای داشبورد
python -m http.server 8080 --directory frontend
# سپس مرورگر → http://localhost:8080
```

### ۲) اجرا با docker-compose (staging)
```bash
cp .env.example .env
# JWT secret تولید کن
echo "MRTO_JWT_SECRET=$(openssl rand -hex 32)" >> .env

# deploy
./deploy/deploy.sh staging

# dashboard → http://localhost:8080
# health     → http://localhost:8080/api/v1/health
```

### ۳) اجرا با Kubernetes (production)
```bash
# apply manifests
./deploy/k8s/deploy.sh apply

# یا با Helm
helm install mrto ./deploy/helm/mrto \
  --set secrets.jwtSecret=$(openssl rand -hex 32) \
  --set secrets.adminPass=$(openssl rand -base64 18) \
  --set ingress.host=mrto.your-domain.com

# status
./deploy/k8s/deploy.sh status
```

---

## 🔁 خط لوله CI/CD

فایل `.github/workflows/ci.yml` پنج مرحله (job) دارد:

| Job | Trigger | وظیفه |
|-----|---------|------|
| **test** | هر push/PR | lint (ruff) + security lint (bandit) + آموزش مدل‌ها + pytest با coverage ≥ 70% |
| **build-and-push** | push به main/develop | build multi-arch (amd64+arm64)، push به GHCR، cosign keyless signing، provenance attestation |
| **security-scan** | بعد از build | Trivy scan، آپلود SARIF به GitHub Security tab، fail روی CRITICAL |
| **deploy-staging** | push به main | compose up + smoke test (health + auth) |
| **release** | tag `v*.*.*` | ایجاد GitHub Release با artifacts: models tarball + SBOM + BRS/SRS PDF |

### تریگرها
- **push به main/develop** → test → build → scan → (staging deploy روی main)
- **Pull request به main** → test
- **tag `v1.0.0`** → test → build → scan → release با artifacts
- **`workflow_dispatch`** → اجرای دستی با گزینه `skip_deploy`

### محدودیت‌های امنیتی
- `permissions` least-privilege (هر job فقط scope لازم)
- `GITHUB_TOKEN` خودکار برای GHCR (بدون PAT)
- cosign با OIDC (keyless، بدون کلید بلندمدت)
- coverage gate (≥ 70%)، Trivy gate (no CRITICAL)

### اجرای محلی CI
برای شبیه‌سازی محلی:
```bash
# همان مراحل CI
cd backend
python -m mrto.synthetic_data data/raw
python -m mrto.train_models --out-dir data --models-dir models --report training_report.json
python -m pytest tests -v --cov=mrto --cov-report=term-missing
```

---

## 📊 داده‌ها

### داده مصنوعی فیزیک‌محور
با `synthetic_data.py` داده‌های واقع‌گرا بر اساس فیزیک فرآیند ریفورمینگ تولید می‌شوند:
- مدل خزش **Larson-Miller** برای محاسبه عمر آلیاژ HP-Modified
- توزیع دمای دیواره لوله‌ها تحت تأثیر مشعل‌ها، خوراک و نویز
- رابطه S/C ↔ بازده و ریسک کک
- ۳۲۰ لوله × ۲۴ ساعت × ۳۰ دقیقه sampling → ۷۶٬۸۰۰ رکورد دما

### داده واقعی صنعتی
از پایگاه‌های عمومی UCI برای اعتبارسنجی مدل‌ها استفاده می‌شود:
- **Airfoil Self-Noise** (NASA) — برای ماژول پایش دما
- **Combined Cycle Power Plant (CCPP)** — برای ماژول بهینه‌سازی بازده

---

## 📈 نتایج آموزش (TRL 5)

| ماژول | معیار | مقدار | هدف SRS | وضعیت |
|-------|------|-------|---------|------|
| ۱. پایش دما | RMSE (°C) | ۳٫۰۹ | ≤ ۱۵ | ✅ |
| ۱. پایش دما | R² | ۰٫۹۴۷ | ≥ ۰٫۸ | ✅ |
| ۲. RUL | MAE (h) | ۶٬۹۵۹ | ≤ ۱۰٬۰۰۰ | ✅ |
| ۲. RUL | AUC ۹۰d | ۱٫۰۰۰ | ≥ ۰٫۸۵ | ✅ |
| ۳. بهینه‌سازی | خطای نسبی | ۰٫۸٪ | ≤ ۵٪ | ✅ |
| ۳. بهینه‌سازی | R² بازده | ۰٫۹۹۱ | ≥ ۰٫۷ | ✅ |
| ۴. مشعل/کاتالیست | AUC | ۱٫۰۰۰ | ≥ ۰٫۹۰ | ✅ |

**همه ۶ هدف کیفی SRS برآورده شد.** جزئیات: `training_report.json`

---

## 🧪 تست‌ها
```bash
cd backend
python -m pytest tests -v --cov=mrto --cov-report=term-missing
```
- **۴۸ تست** (unit + integration + ML + API)
- **پوشش ۷۴٪** کل ماژول‌ها

---

## 📁 ساختار پروژه

```
inno-marjan-petrochemical-/
├── .github/workflows/ci.yml       # 🔁 خط لوله CI/CD (5 job)
├── .env.example                   # 🔧 template تنظیمات
├── docker-compose.yml             # 🐳 استک محلی/staging
├── deploy/
│   ├── nginx.conf                 # ⚙️ reverse proxy + rate-limit + security headers
│   ├── deploy.sh                  # 🚀 helper compose up/down
│   ├── k8s/
│   │   ├── manifests.yaml         # ⎈ K8s manifests (deployment, svc, ingress, HPA, PDB, NetPol)
│   │   └── deploy.sh              # ⎈ helper kubectl apply/rollback
│   └── helm/mrto/                 # ⎈ Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   └── mrto/
│       ├── app.py                 # FastAPI + JWT + RBAC + audit
│       ├── db.py                  # SQLite
│       ├── models.py              # 4 inference wrappers
│       ├── synthetic_data.py      # physics-based generator
│       ├── real_data.py           # UCI loader
│       └── train_models.py        # training pipeline
├── frontend/
│   └── index.html                 # responsive dashboard
├── data/raw/                      # synthetic + UCI data
├── models/                        # trained sklearn models (.joblib)
├── docs/                          # BRS + SRS (docx + pdf)
├── tests/                         # traceability matrix
├── training_report.json           # quality metrics
├── TRL5_ASSESSMENT.md             # TRL5 evaluation
├── SECURITY.md                    # vulnerability disclosure + security features
└── README.md                      # این فایل
```

---

## 🛡 امنیت

برای گزارش آسیب‌پذیری و سیاست امنیتی به [SECURITY.md](SECURITY.md) مراجعه کنید.

ویژگی‌های امنیتی:
- ✅ JWT auth (HS256, 256-bit secret)
- ✅ RBAC سه‌سطحی (admin/operator/viewer)
- ✅ Audit log
- ✅ bcrypt password hashing (12 rounds)
- ✅ Rate limiting در nginx
- ✅ Container image **cosign-signed** + **provenance attestation** + **SBOM**
- ✅ Trivy scan در CI
- ✅ Non-root containers (UID 1000)
- ✅ NetworkPolicy restricts backend ingress

---

## 📜 مجوز
این پروژه تحت مجوز MIT منتشر شده است.

## 👥 توسعه‌دهندگان
- **پتروشیمی مرجان** — کارفرما
- **شرکت پتروپالاسوس (Petropallasus)** — توسعه‌دهنده

## 📚 مستندات
- [BRS — مستند کسب‌واره](docs/MRTO_BRS.pdf)
- [SRS — مستند نیازمندی‌های نرم‌افزار](docs/MRTO_SRS.pdf)
- [TRL5 ارزیابی سطح آمادگی فناوری](TRL5_ASSESSMENT.md)
- [SECURITY.md — سیاست امنیتی](SECURITY.md)
