# Deploy CortexOS Demo on Vercel

Follow these steps to deploy the demo so it runs correctly and talks to your CortexOS API on EC2.

---

## Prerequisites

- CortexOS API running on EC2 (see **HTTPS requirement** below).
- Your CortexOS repo pushed to **GitHub** (including the `Demo_Project` folder).

---

## Important: API must use HTTPS when demo is on Vercel

The demo on Vercel is served over **HTTPS**. Browsers **block** requests from an HTTPS page to an **HTTP** API (mixed content). So you will see **"Failed to fetch"** if the API URL is `http://3.87.235.87:8000`.

**Fix:** Expose your CortexOS API over **HTTPS** and use that URL in the demo.

**Option A – Nginx + SSL on EC2 (recommended)**  
1. Install nginx and get an SSL certificate (e.g. Let’s Encrypt with certbot, or use a domain pointing to your EC2 IP).  
2. Configure nginx as a reverse proxy: HTTPS (443) → `http://127.0.0.1:8000`.  
3. In the demo, set API URL to `https://your-domain.com` (or `https://your-ec2-domain`).

**Option B – Tunnel (quick test)**  
Use a tunnel that gives you an HTTPS URL to your local/EC2 port 8000 (e.g. [ngrok](https://ngrok.com) or [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps)).  
Then set API URL in the demo to the tunnel’s `https://...` URL.

**Option C – Demo and API on same origin**  
If you host the demo on the same EC2 (e.g. nginx serving the demo and proxying `/api` to the CortexOS app), same-origin requests avoid mixed content; you can use a relative API URL.

---

## Step 1: Open Vercel

1. Go to **[vercel.com](https://vercel.com)** and sign in (GitHub login is easiest).
2. Click **Add New…** → **Project**.

---

## Step 2: Import your repo

1. Select your **GitHub** account and find the **CortexOS** repository.
2. Click **Import** next to it.
3. Do **not** click Deploy yet—configure first.

---

## Step 3: Set Root Directory

1. Under **Configure Project**, find **Root Directory**.
2. Click **Edit** and set it to: **`Demo_Project`**
3. This makes Vercel deploy only the demo app (not the whole CortexOS backend).

---

## Step 4: Build settings (keep defaults)

- **Framework Preset:** Other (or leave as detected).
- **Build Command:** Leave **empty** (this is a static site, no build step).
- **Output Directory:** Leave **empty** or **`.`** (Vercel will serve the root).
- **Install Command:** Leave empty.

Click **Deploy**.

---

## Step 5: Wait for deploy

- Vercel will build and deploy (usually under a minute).
- When done, you’ll get a URL like **`https://cortexos-demo-xxx.vercel.app`**.

---

## Step 6: Test the demo

1. Open the Vercel URL. You’ll be redirected to the **chat** page.
2. The demo is pre-configured with **API URL** = `http://3.87.235.87:8000` (your EC2 API).
3. Click **Save** (to ensure settings are stored), then **Check connection**.
   - You should see **Connected** (green).
4. Send a message in the chat. It will ingest and query your CortexOS API.
5. Use **Load timeline** and **Load graph** (with an entity name) to test those too.

---

## If your API URL is different

- If your CortexOS API is at another URL (e.g. different EC2 IP or a domain):
  1. On the demo page, change **API URL** to that URL.
  2. Click **Save**.
  3. The value is stored in the browser (localStorage) and will be used for all requests.

---

## Optional: Custom domain

- In the Vercel project: **Settings** → **Domains** → add your domain and follow the DNS instructions.

---

## Summary

| What | Value |
|------|--------|
| Root Directory | `Demo_Project` |
| Build Command | (empty) |
| Output Directory | (empty) or `.` |
| API URL (default in app) | `http://3.87.235.87:8000` |

After deploy, the demo runs entirely in the browser and calls your EC2 API. CORS is already enabled on CortexOS, so it works without extra backend changes.
