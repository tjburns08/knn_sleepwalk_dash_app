// assets/knn.js
(function () {
  const w = window;
  w.dash_clientside = Object.assign({}, w.dash_clientside, {
    knn: {
      update_on_hover: function (hover1, hover2, k, knnLeft, knnRight, last, xy) {
        const cs = w.dash_clientside;
        const hasPatch = !!(cs && cs.make_patch);

        // Helpers
        function P(obj) { return cs.make_patch(obj); }
        function emptyPatch() { return P({ data: { 1: { x: [], y: [] }, 2: { x: [], y: [] } } }); }
        function noUpdTriplet(x = last || { idx: null, k: null }) {
          return [cs.no_update, cs.no_update, x];
        }
        function figFull(xy, allIdx, centerIdx, title) {
          return {
            data: [
              {type:'scattergl', mode:'markers', x: xy.x, y: xy.y,
               marker:{size:4, opacity:0.35}, hovertemplate:'<extra></extra>', showlegend:false},
              {type:'scattergl', mode:'markers',
               x: allIdx.map(i=>xy.x[i]), y: allIdx.map(i=>xy.y[i]),
               marker:{size:8, opacity:0.95}, hoverinfo:'skip', showlegend:false},
              {type:'scattergl', mode:'markers',
               x:[xy.x[centerIdx]], y:[xy.y[centerIdx]],
               marker:{size:12, opacity:1}, hoverinfo:'skip', showlegend:false}
            ],
            layout:{title, height:520, margin:{l:10,r:10,t:40,b:10},
                    uirevision:'keep', dragmode:'pan', hovermode:'closest'}
          };
        }

        // If we don't even have xy yet, just clear overlays (patch) or no-op (fallback)
        if (!xy || !Array.isArray(xy.x) || xy.x.length === 0) {
          return hasPatch ? [emptyPatch(), emptyPatch(), { idx:null, k:null }] : noUpdTriplet({ idx:null, k:null });
        }

        // Who triggered?
        const trig = (cs.callback_context.triggered || []).map(t => t.prop_id.split(".")[0])[0];
        let idx = null;
        if (trig === "plot1" && hover1 && hover1.points && hover1.points[0]) {
          idx = hover1.points[0].pointIndex;
        } else if (trig === "plot2" && hover2 && hover2.points && hover2.points[0]) {
          idx = hover2.points[0].pointIndex;
        } else if (last && last.idx != null) {
          idx = last.idx;
        }

        // No valid index yet: DO NOT blank the figure—just don't update
        if (idx == null || idx >= xy.x.length) {
          return hasPatch ? [emptyPatch(), emptyPatch(), { idx:null, k:null }] : noUpdTriplet({ idx:null, k:null });
        }

        // Clamp k
        k = Math.max(1, parseInt(k || 1, 10));
        const kAvail = Math.min((knnLeft[idx]||[]).length, (knnRight[idx]||[]).length);
        k = Math.min(k, kAvail);

        // Skip redundant work
        if (last && last.idx === idx && last.k === k) {
          return noUpdTriplet(last);
        }

        const leftAll  = [idx].concat((knnLeft[idx]  || []).slice(0, k));
        const rightAll = [idx].concat((knnRight[idx] || []).slice(0, k));

        if (hasPatch) {
          const leftPatch  = P({ data: { 1: { x: leftAll.map(i=>xy.x[i]),  y: leftAll.map(i=>xy.y[i]) },
                                         2: { x: [xy.x[idx]],              y: [xy.y[idx]] } } });
          const rightPatch = P({ data: { 1: { x: rightAll.map(i=>xy.x[i]), y: rightAll.map(i=>xy.y[i]) },
                                         2: { x: [xy.x[idx]],              y: [xy.y[idx]] } } });
          return [leftPatch, rightPatch, { idx, k }];
        } else {
          // Full-figure fallback (no patch API)
          const leftFig  = figFull(xy, leftAll,  idx, 'UMAP layout — kNN in UMAP space');
          const rightFig = figFull(xy, rightAll, idx, 'UMAP layout — kNN in feature space');
          return [leftFig, rightFig, { idx, k }];
        }
      }
    }
  });
})();
