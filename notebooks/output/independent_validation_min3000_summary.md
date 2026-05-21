# Tong hop ket qua Qasper independent variants

Ngay tong hop: 2026-05-21

## 1. Boi canh thi nghiem

Tat ca output trong thu muc nay la ket qua chay cac bien the RAG doc lap tren Qasper validation voi cau hinh chung:

- Split: `validation`
- Loc tai lieu dai: `MIN_DOC_WORDS=3000`
- So document sau loc: `165`
- So QA examples: `583`
- Retriever dense mac dinh: `sentence-transformers/all-MiniLM-L6-v2`
- Generator mac dinh: `google/flan-t5-base`
- `top_k=5`, `retrieve_k=20`, `chunk_size=180`, `overlap=40`

Bo survey do dai tai lieu cho thay Qasper validation co 281 documents va 1005 QA examples. Nguong `>=3000` words giu lai 165 documents, tuong duong 58.7% documents va 58.0% QA examples. Vi vay day la tap long-document kha hop ly de so sanh RAG, nhung khong phai benchmark ultra-long-context thuan tuy; cac nguong `>=8000` va `>=12000` words chi con lan luot 30 va 14 QA examples.

## 2. Bang ket qua chinh

| Variant | Token F1 | Answer Recall@5 | Context Recall | Context Precision | Faithfulness | Relevancy | Runtime | Sec/example | Doc/QA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `semantic_chunking_dense` | **0.2256** | 0.1561 | 0.3279 | 0.5300 | 0.3722 | **0.1356** | 221.20s | 0.379s | 165/583 |
| `raptor_leiden_abstractive` | 0.1522 | 0.1787 | 0.3521 | 0.4868 | 0.4648 | 0.0864 | 1655.36s | 2.839s | 165/583 |
| `raptor_extractive` | 0.1425 | 0.1838 | 0.3719 | 0.5743 | 0.4803 | 0.0755 | 3043.50s | 5.220s | 165/583 |
| `dense_reranker` | 0.1190 | **0.2207** | **0.4603** | **0.6518** | 0.5506 | 0.0456 | 210.81s | 0.362s | 165/583 |
| `base_dense` | 0.1165 | 0.2018 | 0.4147 | 0.6021 | 0.5472 | 0.0411 | 179.79s | 0.308s | 165/583 |
| `dense_recency_heavy` | 0.1047 | 0.2018 | 0.4147 | 0.6021 | 0.5146 | 0.0426 | 169.66s | 0.291s | 165/583 |
| `dense_u_shape` | 0.1043 | 0.2018 | 0.4147 | 0.6021 | 0.5232 | 0.0407 | 177.60s | 0.305s | 165/583 |
| `hybrid_rrf` | 0.1021 | 0.2053 | 0.4280 | 0.6343 | **0.5643** | 0.0397 | 186.96s | 0.321s | 165/583 |
| `bm25_only` | 0.0891 | 0.1795 | 0.3504 | 0.6153 | 0.5437 | 0.0293 | **145.84s** | **0.250s** | 165/583 |

## 3. Xep hang theo muc tieu

Neu muc tieu la answer quality theo metric lexical:

1. `semantic_chunking_dense`: cao nhat ve Token F1 va Answer Relevancy.
2. `raptor_leiden_abstractive`: F1 cao hon baseline, nhung rat cham.
3. `raptor_extractive`: F1 cao hon baseline, nhung cham nhat.

Neu muc tieu la retrieval-chat luong RAG co the bao ve trong nghien cuu:

1. `dense_reranker`: cao nhat ve Answer Recall@5, Context Recall, Context Precision.
2. `hybrid_rrf`: tang Context Precision va Faithfulness so voi baseline, nhung F1 khong tang.
3. `base_dense`: baseline on dinh, dung lam moc so sanh tot.

Neu muc tieu la runtime:

1. `bm25_only`: nhanh nhat, nhung F1 va relevancy thap.
2. `dense_recency_heavy`: nhanh hon base_dense mot chut, nhung khong cai thien chat luong.
3. `base_dense`: can bang hon BM25-only.

## 4. Nhan xet tung nhom phuong phap

### `base_dense`

Baseline dense retrieval on dinh va cho retrieval kha tot: Context Recall 0.4147, Precision 0.6021. Tuy nhien Token F1 chi 0.1165 va Relevancy 0.0411, cho thay retrieval co ich nhung SLM/generator chua bien context thanh cau tra loi tot.

### `bm25_only`

BM25-only nhanh nhat, Context Precision 0.6153 khong te, nhung Context Recall 0.3504 va F1 0.0891 thap nhat. Ket luan: sparse keyword retrieval khong du lam main method, nhung co the dung lam thanh phan phu trong fusion.

### `dense_u_shape` va `dense_recency_heavy`

Hai bien the reorder context khong cai thien so voi baseline. Vi retrieval metrics giong baseline nhung F1/faithfulness giam, dau hieu hien tai la viec sap xep lai context khong giup FLAN-T5-base trong cau hinh nay.

### `hybrid_rrf`

Hybrid RRF tang Answer Recall@5 tu 0.2018 len 0.2053, Context Recall tu 0.4147 len 0.4280, Context Precision tu 0.6021 len 0.6343, va Faithfulness cao nhat 0.5643. Diem yeu la F1 giam xuong 0.1021, nen loi ich retrieval chua duoc generator khai thac.

### `dense_reranker`

Dense reranker la bien the RAG thuyet phuc nhat neu uu tien retrieval: Answer Recall@5 0.2207, Context Recall 0.4603, Context Precision 0.6518 deu cao nhat. Runtime 210.81s van chap nhan duoc, chi cham hon base_dense khoang 31s tren 583 examples. Diem can fix tiep theo khong nam chu yeu o retrieval, ma o generator/prompting/answer extraction.

### `semantic_chunking_dense`

Semantic chunking dense thang ro ve Token F1 0.2256 va Relevancy 0.1356, gan gap doi baseline. Nhung Context Recall 0.3279, Precision 0.5300, Faithfulness 0.3722 lai thap. Prediction-level sample cho thay no co xu huong sinh cau tra loi ngan, doi khi gan lexical hon voi gold answer, nhung evidence grounding kem hon. Bien the nay can duoc inspect ky de tranh ket luan dua tren artifact cua metric lexical.

### RAPTOR variants

`raptor_extractive` va `raptor_leiden_abstractive` deu tang F1 so voi baseline, nhung chi phi runtime qua cao. `raptor_extractive` mat 3043.50s, cham khoang 16.9x base_dense; `raptor_leiden_abstractive` mat 1655.36s, cham khoang 9.2x. Retrieval va faithfulness khong vuot reranker. Hien tai chua nen chon RAPTOR lam huong chinh neu muc tieu la method hieu qua/ti le cost-performance tot.

## 5. Kiem tra prediction-level nhanh

Tu cac file prediction hien co:

| Variant | Examples | Unanswerable | Unanswerable rate | F1 > 0 | F1 > 0 rate | Recall@5 > 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_dense` | 583 | 305 | 0.52 | 163 | 0.28 | 162 |
| `bm25_only` | 583 | 310 | 0.53 | 128 | 0.22 | 144 |
| `dense_recency_heavy` | 583 | 287 | 0.49 | 153 | 0.26 | 162 |
| `dense_reranker` | 583 | 302 | 0.52 | 164 | 0.28 | 178 |
| `dense_u_shape` | 583 | 293 | 0.50 | 138 | 0.24 | 162 |
| `hybrid_rrf` | 583 | 314 | 0.54 | 133 | 0.23 | 164 |
| `raptor_leiden_abstractive` | 583 | 208 | 0.36 | 201 | 0.34 | 141 |
| `semantic_chunking_dense` | 583 | 132 | 0.23 | 288 | 0.49 | 122 |

Ghi chu: trong thu muc local hien tai co `raptor_extractive_validation_min3000_summary.json` nhung khong thay `raptor_extractive_validation_min3000_predictions.jsonl`, nen chua co thong ke prediction-level cho `raptor_extractive`.

Nhan xet tu bang phu:

- `semantic_chunking_dense` it tra `Unanswerable` hon nhieu va co ti le F1 > 0 cao nhat. Day giai thich vi sao F1 cao, nhung khong chung minh grounding tot.
- `dense_reranker` co so cau Recall@5 > 0 cao nhat trong cac bien the co prediction file, khop voi ket qua retrieval metrics.
- Baseline va reranker co ty le F1 > 0 gan nhau, nghia la retrieval tot hon cua reranker chua duoc generator chuyen thanh answer F1 tuong ung.

## 6. Ket luan de viet vao bao cao

Ket qua hien tai ung ho mot ket luan hai lop:

1. Ve metric answer lexical, `semantic_chunking_dense` dang dung dau. Tuy nhien no co context recall/precision va faithfulness thap, nen can phan tich loi theo tung prediction truoc khi tuyen bo day la method tot nhat.
2. Ve RAG pipeline co grounding/retrieval defensible, `dense_reranker` la ung vien manh nhat de phat trien tiep. No cai thien retrieval ro rang, runtime van hop ly, va that bai chinh nam o generation/answer extraction.

Huong tiep theo nen chon:

- Neu can cau chuyen nghien cuu chac chan: chon `dense_reranker`, sau do tune generator prompt, answer extraction, threshold Unanswerable, va co the ket hop semantic chunking sau khi da kiem soat faithfulness.
- Neu can toi uu diem F1 truoc: inspect 50-100 predictions cua `semantic_chunking_dense` de xac dinh F1 cao la cai thien that hay artifact, roi moi quyet dinh co theo tiep hay khong.
- Tam thoi khong nen dau tu them vao RAPTOR vi runtime qua cao so voi loi ich hien tai.

