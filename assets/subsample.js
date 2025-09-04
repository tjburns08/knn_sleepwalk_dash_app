// assets/subsample.js
(function () {
  const w = window;
  w.dash_clientside = Object.assign({}, w.dash_clientside, {
    subsample: {
      // Inputs: upload-orig.contents, upload-dimr.contents, max-cells (index)
      // Outputs: subset-orig-csv.data, subset-dimr-csv.data, subset-meta.data
      prepare: function (cOrig, cDimr, capIdx) {
        // If either file missing, clear stores
        if (!cOrig || !cDimr || capIdx == null) {
          return [null, null, null];
        }

        // Same choices as Python
        const CHOICES = [5000, 10000, 20000, 30000, 40000, 50000, 0];
        const cap = CHOICES[Math.max(0, Math.min(CHOICES.length-1, parseInt(capIdx,10)))];

        // Helpers
        function decodeDataURL(dataURL) {
          // "data:text/csv;base64,XXXX"
          const b64 = String(dataURL).split(",", 2)[1] || "";
          // numeric CSV → latin1 decode is fine
          const bin = atob(b64);
          return bin;
        }
        function splitLines(raw) {
          // Normalize newlines and drop final empty line if any
          return raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
        }
        function seededPick(n, k, seed) {
          // Simple deterministic subset without replacement (Mulberry32 + Fisher-Yates)
          function mulberry32(a){return function(){let t=a+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;}}
          const rnd = mulberry32(seed>>>0);
          const idx = Array.from({length:n}, (_,i)=>i);
          // partial shuffle
          for (let i=n-1; i>n-1-k; --i) {
            const j = Math.floor(rnd()*(i+1));
            [idx[i], idx[j]] = [idx[j], idx[i]];
          }
          return idx.slice(n-k).sort((a,b)=>a-b);
        }
        function encodeCSVString(lines) {
          return lines.join("\n");
        }

        // Decode both CSVs (as raw text)
        const rawOrig = decodeDataURL(cOrig);
        const rawDimr = decodeDataURL(cDimr);

        // Split to lines
        let linesO = splitLines(rawOrig);
        let linesD = splitLines(rawDimr);

        // Drop trailing empty line
        if (linesO.length && linesO[linesO.length-1] === "") linesO.pop();
        if (linesD.length && linesD[linesD.length-1] === "") linesD.pop();

        if (linesO.length === 0 || linesD.length === 0) {
          console.error("Empty CSV after decode.");
          return [null, null, null];
        }

        const headerO = linesO[0];
        const headerD = linesD[0];
        const dataO = linesO.slice(1);
        const dataD = linesD.slice(1);

        const nO = dataO.length;
        const nD = dataD.length;
        const n = Math.min(nO, nD); // force alignment by order

        if (n <= 0) {
          console.error("No data rows.");
          return [null, null, null];
        }

        // If "No subsample" or already small → return originals (aligned to min rows)
        const wanted = (cap === 0) ? n : Math.min(n, cap);
        let keepIdx;
        if (wanted === n) {
          // No subsample: keep first n rows (order-aligned)
          keepIdx = Array.from({length:n}, (_,i)=>i);
        } else {
          // Uniform random subset with fixed seed (42)
          keepIdx = seededPick(n, wanted, 42);
        }

        // Build new CSVs with same row selection for both tables
        const subO = [headerO].concat(keepIdx.map(i => dataO[i])).join("\n");
        const subD = [headerD].concat(keepIdx.map(i => dataD[i])).join("\n");

        const meta = { n_in: n, n_out: wanted, cap: cap, method: "uniform" };

        return [subO, subD, meta];
      }
    }
  });
})();
