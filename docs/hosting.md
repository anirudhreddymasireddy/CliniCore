# Hosting CliniCore: Exasol SaaS + a cloud machine

Goal: a public link that works on any phone or laptop, with the data in Exasol SaaS.

```
phone / laptop ──https──► Cloudflare quick tunnel ──► cloud machine (AWS sandbox): CliniCore app ──► Exasol SaaS
```

Never put the Exasol token or the app password in a file inside the repository or in a chat.

## Part A · Exasol SaaS (on your laptop, ~15 min)

1. Start the free Exasol SaaS trial (exasol.com → *Try Exasol SaaS*; no credit card).
2. Create a database: smallest size. If you can choose, pick an AWS region close to the cloud machine (the AWS sandbox runs in us-west-2 / us-east-1).
3. Open **Connect** for the database. The wizard asks for an IP address to allow: add your laptop's public IP (open https://checkip.amazonaws.com in the browser). Copy:
   - the connection string (host) and port 8563
   - your user name
   - the **personal access token** (shown only once): save it to `%USERPROFILE%\exasol_saas_pat.txt`, outside the repo.
4. Copy the app's tables from your local Exasol to SaaS:

```powershell
cd "$HOME\Downloads\CliniCore"
# 1) export from the LOCAL database
$env:EXA_DSN="127.0.0.1:8563"; $env:EXA_USER="sys"
$env:EXA_PASS=(Get-Content "$HOME\.exasol-starter-kit\credentials\personal_sys_password" -Raw).Trim()
python scripts\export_exasol.py CAREGUARD_RESULTS --out exasol_export

# 2) import into SaaS
$env:EXA_DSN="<connection string>:8563"; $env:EXA_USER="<your SaaS user>"
$env:EXA_PASS=(Get-Content "$HOME\exasol_saas_pat.txt" -Raw).Trim()
python scripts\import_exasol.py exasol_export
python scripts\selftest_exasol.py          # expect 4 x PASS
```

`exasol_export/` is in `.gitignore`. Optional: also copy `CAREGUARD_SEPSIS19` (1.5 M rows, slower) if you want to retrain on SaaS.

## Part B · AWS sandbox (VS Code in the browser)

1. Open the event's VS Code URL → menu → *Terminal* → *New Terminal*.
2. Check the machine and get its public IP:

```bash
python3 --version; nproc; free -h; curl -s https://checkip.amazonaws.com
```

3. Add that IP to the Exasol SaaS allow list (web console → Security / network access).
4. Get the code. For a private repository, create a GitHub token (github.com → Settings → Developer settings → Fine-grained tokens → read access to CliniCore) and paste it when git asks for the password:

```bash
git clone https://github.com/anirudhreddymasireddy/CliniCore.git && cd CliniCore
```

5. Start it (the password protects the public link; user name can be anything):

```bash
read -s -p "App password: " CLINICORE_PASSWORD; export CLINICORE_PASSWORD; echo
read -s -p "Exasol SaaS token: " EXA_PASS; export EXA_PASS; echo
export CLINICORE_SOURCE=exasol EXA_DSN='<connection string>:8563' EXA_USER='<your SaaS user>'
bash scripts/cloud_up.sh
```

It installs the Python packages, checks the database, downloads the Cloudflare client and prints `CliniCore is live at: https://….trycloudflare.com`.

Without Exasol (fallback, works offline from the database): `unset CLINICORE_SOURCE; bash scripts/cloud_up.sh` uses the result files committed in the repo.

## Notes

- Stop: `bash scripts/cloud_down.sh`. Logs: `app.log`, `tunnel.log`.
- The quick-tunnel link changes every time it restarts and has no uptime guarantee; fine for a demo.
- The AWS sandbox ends with the event (about Monday 16 Nov, 11:00); the SaaS trial keeps the data after that.
- If the venue Wi-Fi IP changes, add the new one to the SaaS allow list.
- SaaS databases can stop automatically when idle; start it in the web console before the demo.
- The connection to SaaS checks the server's TLS certificate. If the check fails with a certificate error on a machine with an old certificate store, `export EXA_INSECURE=1` turns the check off (still encrypted, but not verified); use it only to get unblocked.
