using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Debug = UnityEngine.Debug;
using ReadOnlyAttribute = Unity.Collections.ReadOnlyAttribute;
using WriteOnlyAttribute = Unity.Collections.WriteOnlyAttribute;

namespace Chisel.Core
{
    struct SubMeshDescriptions
    {
        public MeshQuery meshQuery;
        public int meshQueryIndex;
        public int subMeshQueryIndex;
        public ulong surfaceParameter;
        public uint  chunk;
            
        public int surfacesOffset;
        public int surfacesCount;
            
        public int vertexCount;
        public int indexCount;
            
        public uint geometryHashValue;  // used to detect changes in vertex positions  
        public uint surfaceHashValue;   // used to detect changes in color, normal, tangent or uv (doesn't effect lighting)
    }

	struct SubMeshSurface
	{
		public CompactNodeID brushNodeID;
		public int surfaceIndex;
		public ulong surfaceParameter;
		public uint  chunk;

		public int vertexCount;
		public int indexCount;

		public uint geometryHashValue;
		public uint surfaceHashValue;
		public BlobAssetReference<ChiselBrushRenderBuffer> brushRenderBuffer;
	}

	// What a query's surfaces are grouped by: their parameter, and for a collider the chunk of space its brush is in
	struct SurfaceGroupKey : IComparable<SurfaceGroupKey>
	{
		public ulong parameter;
		public uint  chunk;
		public int CompareTo(SurfaceGroupKey other)
		{
			if (parameter != other.parameter) return parameter < other.parameter ? -1 : 1;
			if (chunk != other.chunk) return chunk < other.chunk ? -1 : 1;
			return 0;
		}
		public bool Equals(SurfaceGroupKey other) => parameter == other.parameter && chunk == other.chunk;
	}

	struct BrushData
	{
		public IndexOrder brushIndexOrder; //<- TODO: if we use NodeOrder maybe this could be explicit based on the order in array?
		public int brushSurfaceOffset;
		public int brushSurfaceCount;

		public BlobAssetReference<ChiselBrushRenderBuffer> brushRenderBuffer;
	}


	[BurstCompile(CompileSynchronously = true)]
    struct AllocateSubMeshesJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public int                  meshQueryLength;
        [NoAlias, ReadOnly] public NativeReference<int> surfaceCountRef;

        // Read/Write
        [NoAlias] public NativeList<SubMeshDescriptions> subMeshDescriptions;
        [NoAlias] public NativeList<SubMeshSection>      subMeshSections;

        public void Execute()
        {
            var subMeshCapacity = surfaceCountRef.Value * meshQueryLength;
            if (subMeshDescriptions.Capacity < subMeshCapacity)
                subMeshDescriptions.Capacity = math.max(4, subMeshCapacity);
            
            if (subMeshSections.Capacity < subMeshCapacity)
                subMeshSections.Capacity = math.max(4, subMeshCapacity);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    struct FindBrushRenderBuffersJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public int meshQueryLength;
        [NoAlias, ReadOnly] public NativeList<IndexOrder>                                  allTreeBrushIndexOrders;
        [NoAlias, ReadOnly] public NativeList<BlobAssetReference<ChiselBrushRenderBuffer>> brushRenderBufferCache;
        // the brushes whose buffers this update made again
        [NoAlias, ReadOnly] public NativeList<IndexOrder>                                  allUpdateBrushIndexOrders;

        // Write
        [NoAlias] public NativeList<BrushData> brushRenderData;

        [NoAlias] public NativeReference<int> surfaceCountRef;
        // what the weld did at the last update, kept with the tree
        [NoAlias] public WeldMemory weldMemory;
        // where the weld's last step moved the corners it moved (OutputModelWeld.WeldModel), kept with the tree
        [NoAlias] public NativeList<float3x2> collapseMoves;
        // the surfaces the weld changed and the brushes to store again (CopyWeldedBrushesJob)
        [NoAlias] public WeldState weld;

        public void Execute()
        {
            var totalSurfaceCount = 0;
            for (int b = 0, count_b = allTreeBrushIndexOrders.Length; b < count_b; b++)
            {
                var brushIndexOrder     = allTreeBrushIndexOrders[b];
                var brushNodeOrder      = allTreeBrushIndexOrders[b].nodeOrder;
                var brushRenderBuffer   = brushRenderBufferCache[brushNodeOrder];
                if (!brushRenderBuffer.IsCreated)
                    continue;

                ref var brushRenderBufferRef = ref brushRenderBuffer.Value;

                var brushSurfaceCount = brushRenderBufferRef.surfaceCount;
                if (brushSurfaceCount == 0)
                    continue;

                var brushSurfaceOffset = brushRenderBufferRef.surfaceOffset;
                totalSurfaceCount += brushSurfaceCount;
                brushRenderData.AddNoResize(new BrushData{
                    brushIndexOrder     = brushIndexOrder,
                    brushSurfaceOffset  = brushSurfaceOffset,
                    brushSurfaceCount   = brushSurfaceCount,
                    brushRenderBuffer   = brushRenderBuffer
                });
            }
            if (surfaceCountRef.Value < totalSurfaceCount)
                surfaceCountRef.Value = totalSurfaceCount;

            // The weld at the very end across the whole model: needles that close a seam go with the triangle across them
            // (its collapse runs brush by brush in the jobs after this one; the brushes it changed are stored again by
            // CopyWeldedBrushesJob)
            OutputModelWeld.PrepareIncremental(brushRenderData, allUpdateBrushIndexOrders.AsArray(), ref weldMemory, ref weld);
            do
                OutputModelWeld.PlanWeldBeforeCollapse(brushRenderData, ref weld, out _, out _, out _, collapseMoves);
            while (OutputModelWeld.TakeInSplitSurfaces(brushRenderData, ref weldMemory, ref weld));
        }
    }

    // The collapse of the weld, brush by brush: each brush's short edges counted (write false), then written where
    // OffsetShortEdges puts them
    [BurstCompile(CompileSynchronously = true)]
    unsafe struct ShortEdgesJob : IJobParallelForDefer
    {
        public bool write;
        [NoAlias, ReadOnly] public NativeArray<BrushData>              brushRenderData;
        [NoAlias, ReadOnly] public NativeList<OutputModelWeld.Working> work;
        [NoAlias, ReadOnly] public NativeParallelHashMap<long, int>    workIndex;
        [NoAlias, ReadOnly] public NativeArray<int2>                   edgeOffsets;
        [NoAlias, ReadOnly] public WeldScope                           scope;

        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<int2>     edgeCounts;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<float3x2> shortPairs;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<float3>   keptCorners;

        public void Execute(int b)
        {
            if (!write)
            {
                edgeCounts[b] = OutputModelWeld.ShortEdgesOf(b, brushRenderData, work, workIndex, scope, null, null);
                return;
            }
            OutputModelWeld.ShortEdgesOf(b, brushRenderData, work, workIndex, scope,
                                         (float3x2*)shortPairs.GetUnsafePtr() + edgeOffsets[b].x, (float3*)keptCorners.GetUnsafePtr() + edgeOffsets[b].y);
        }
    }

    // The collapse's steps that see the whole model: 0 where each brush's short edges go, 1 the clusters, 2 where each
    // brush's decisions go, 3 the decisions carried out, and what the weld did kept (OutputModelWeld.RecordIncremental)
    [BurstCompile(CompileSynchronously = true)]
    struct CollapseStepJob : IJob
    {
        public int step;
        [NoAlias] public NativeList<BrushData>  brushRenderData;
        [NoAlias] public NativeList<float3x2>   collapseMoves;
        [NoAlias] public WeldState              weld;
        [NoAlias] public WeldMemory             weldMemory;

        public void Execute()
        {
            switch (step)
            {
                case 0: OutputModelWeld.OffsetShortEdges(ref weld); break;
                case 1: OutputModelWeld.ClusterShortEdges(ref weld, collapseMoves); break;
                case 2: OutputModelWeld.OffsetDecisions(ref weld); break;
                case 3:
                    OutputModelWeld.ApplyCollapse(brushRenderData, ref weld);
                    OutputModelWeld.RecordIncremental(brushRenderData, ref weldMemory, ref weld, collapseMoves);
                    break;
            }
        }
    }

    // What the collapse does to each brush's surfaces, counted (write false), then written where OffsetDecisions puts it
    [BurstCompile(CompileSynchronously = true)]
    unsafe struct DecideCollapseJob : IJobParallelForDefer
    {
        public bool write;
        [NoAlias, ReadOnly] public NativeArray<BrushData>                 brushRenderData;
        [NoAlias, ReadOnly] public NativeList<OutputModelWeld.Working>    work;
        [NoAlias, ReadOnly] public NativeParallelHashMap<long, int>       workIndex;
        [NoAlias, ReadOnly] public NativeParallelHashMap<float3, float3>  movedTo;
        [NoAlias, ReadOnly] public NativeArray<int>                       decisionOffsets;
        [NoAlias, ReadOnly] public WeldScope                              scope;

        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<int>                              decisionCounts;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<OutputModelWeld.CollapseDecision> decisions;

        public void Execute(int b)
        {
            if (!write)
            {
                decisionCounts[b] = OutputModelWeld.DecideCollapse(b, brushRenderData, work, workIndex, scope, movedTo, null);
                return;
            }
            OutputModelWeld.DecideCollapse(b, brushRenderData, work, workIndex, scope, movedTo,
                                           (OutputModelWeld.CollapseDecision*)decisions.GetUnsafePtr() + decisionOffsets[b]);
        }
    }

    // Marks the sections whose mesh is made from the same as last time (keptSections), remembers what each was made from,
    // and gives each render mesh the hash of what its lightmap layout is made from (sectionLightmapHashes); kept: how
    // many are kept. A query's own material meshes and the colliders are kept; the shadow-only and debug meshes are
    // always made again.
    [BurstCompile(CompileSynchronously = true)]
    unsafe struct FindKeptSectionsJob : IJob
    {
        [NoAlias, ReadOnly] public NativeList<SubMeshSection>                sections;
        [NoAlias, ReadOnly] public NativeList<SubMeshDescriptions>           descriptions;
        [NoAlias, ReadOnly] public NativeArray<UnsafeList<SubMeshSurface>>   subMeshSurfaces;
        public LightmapUVSettings lightmapUVSettings;

        [NoAlias] public NativeParallelHashMap<RenderMeshKey, uint4> hashes;
        [NoAlias] public NativeArray<bool>   keptSections;
        [NoAlias] public NativeArray<uint>   sectionLightmapHashes;
        [NoAlias] public NativeReference<int> kept;

        public void Execute()
        {
            var seen   = new NativeParallelHashSet<RenderMeshKey>(math.max(1, sections.Length), Allocator.Temp);
            var buffer = new UnsafeAppendBuffer(256, 16, Allocator.Temp);
            var count  = 0;
            for (int i = 0; i < sections.Length; i++)
            {
                var section  = sections[i];
                var collider = section.meshQuery.LayerParameterIndex == SurfaceParameterIndex.PhysicsMaterial;
                var render   = section.meshQuery.LayerParameterIndex == SurfaceParameterIndex.RenderMaterial;
                if (render)
                    sectionLightmapHashes[i] = LightmapHash(section, ref buffer);
                if (!hashes.IsCreated ||
                    !(collider || GatherSurfacesJob.OneMeshPerMaterial(section.meshQuery)) ||
                    (render && section.endIndex - section.startIndex != 1))
                    continue;
                // a collider's section is the one description it starts at
                var description = descriptions[section.startIndex];
                buffer.Reset();
                if (render)
                {
                    buffer.Add(description.geometryHashValue);
                    buffer.Add(description.surfaceHashValue);
                    buffer.Add(description.vertexCount);
                    buffer.Add(description.indexCount);
                    buffer.Add(lightmapUVSettings);
                }
                var surfaces = subMeshSurfaces[description.meshQueryIndex];
                for (int s = description.surfacesOffset; s < description.surfacesOffset + description.surfacesCount && s < surfaces.Length; s++)
                {
                    buffer.Add(surfaces[s].brushNodeID);
                    buffer.Add(surfaces[s].surfaceIndex);
                    // a collider is the shape as the weld left it, before any hotspot cut
                    if (collider && surfaces[s].brushRenderBuffer.IsCreated)
                        buffer.Add(surfaces[s].brushRenderBuffer.Value.surfaces[surfaces[s].surfaceIndex].shapeHashValue);
                }
                var hash = xxHash3.Hash128(buffer.Ptr, buffer.Length);
                var key  = new RenderMeshKey { query = (int)section.meshQuery.LayerQuery, surfaceParameter = description.surfaceParameter, chunk = description.chunk };
                if (!seen.Add(key))
                    continue;
                if (hashes.TryGetValue(key, out var last) && last.Equals(hash))
                {
                    keptSections[i] = true;
                    count++;
                }
                hashes[key] = hash;
            }
            if (hashes.IsCreated)
            {
                // the meshes this update does not have any more
                using var keys = hashes.GetKeyArray(Allocator.Temp);
                for (int k = 0; k < keys.Length; k++)
                {
                    if (!seen.Contains(keys[k]))
                        hashes.Remove(keys[k]);
                }
            }
            buffer.Dispose();
            seen.Dispose();
            kept.Value = count;
        }

        // What a render mesh's lightmap layout is made from (ChiselOutputRenderable): each surface's chart, which the
        // CSG made before any hotspot cut, and the layout settings
        uint LightmapHash(SubMeshSection section, ref UnsafeAppendBuffer buffer)
        {
            buffer.Reset();
            buffer.Add(lightmapUVSettings);
            for (int d = section.startIndex; d < section.endIndex; d++)
            {
                var description = descriptions[d];
                var surfaces    = subMeshSurfaces[description.meshQueryIndex];
                for (int s = description.surfacesOffset; s < description.surfacesOffset + description.surfacesCount && s < surfaces.Length; s++)
                {
                    if (!surfaces[s].brushRenderBuffer.IsCreated)
                        continue;
                    ref var surface = ref surfaces[s].brushRenderBuffer.Value.surfaces[surfaces[s].surfaceIndex];
                    var drawn = surface.indices.Length > 0 && surface.renderVertices.Length > 0;
                    buffer.Add(surface.lightmapChart);
                    buffer.Add(drawn ? (int)LightmapUVLayout.ModeOf(surface.outputFlags) : -1);
                    buffer.Add(surface.shapeHashValue);
                    buffer.Add(surfaces[s].surfaceIndex);
                }
            }
            return xxHash3.Hash64(buffer.Ptr, buffer.Length).x;
        }
    }

    // Each brush the weld changed, stored again on its own
    [BurstCompile(CompileSynchronously = true)]
    struct CopyWeldedBrushesJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData>              brushRenderData;
        [NoAlias, ReadOnly] public NativeList<OutputModelWeld.Working> work;
        [NoAlias, ReadOnly] public NativeParallelHashMap<long, int>    workIndex;
        [NoAlias, ReadOnly] public NativeList<OutputModelWeld.CollapseDecision> collapseDecisions;

        [NoAlias] public NativeArray<WeldCopy> copies;

        public void Execute(int index)
        {
            copies[index] = OutputModelWeld.CopyPlanned(copies[index], brushRenderData, work, workIndex, collapseDecisions, default);
        }
    }

    // Who is grouped with whom (MatchHotspotGroupsJob), once the weld's copies are handed out (FinishWeldJob)
    [BurstCompile(CompileSynchronously = true)]
    struct PrepareHotspotGroupsJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public NativeArray<HotspotTileData>   hotspotTiles;
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<HotspotLink>       hotspotLinks;
        [NoAlias, ReadOnly] public NativeArray<float4>            hotspotTileSides;

        // Write
        [NoAlias] public NativeList<BrushData> brushRenderData;
        [NoAlias] public HotspotCutState hotspotCut;
        [NoAlias] public FindState       hotspotFind;
        // what the last update's grouping found, kept with the tree
        [NoAlias] public HotspotGroupMemory hotspotGroupMemory;
        public bool verify;

        public void Execute()
        {
            HotspotGroups.Borders(brushRenderData, hotspotLinks, ref hotspotCut.bordersOf);
            HotspotGroups.PrepareIncremental(brushRenderData, hotspotTiles, hotspotLayouts, hotspotLinks, ref hotspotFind, ref hotspotGroupMemory);
            if (verify)
            {
                var full = FindState.Create(Allocator.Temp);
                HotspotGroups.Prepare(brushRenderData, hotspotTiles, hotspotLayouts, hotspotLinks, ref full);
                if (!HotspotGroups.SameAs(hotspotFind, full, out var difference))
                    Debug.LogError($"Incremental hotspot groups differ from the full grouping ({hotspotGroupMemory.lastDirty.Value} dirty, full: {hotspotGroupMemory.lastFull.Value}): {difference}");
                full.Dispose();
            }
            hotspotFind.inputs.Value = HotspotGroups.InputsKey(hotspotTiles, hotspotLayouts, hotspotTileSides);
        }
    }

    // The weld's copies handed to their brushes (each brush's own buffer kept first), and room for the surfaces' kinds
    [BurstCompile(CompileSynchronously = true)]
    struct FinishWeldJob : IJob
    {
        [NoAlias, ReadOnly] public NativeList<WeldCopy> weldCopies;

        [NoAlias] public NativeList<BrushData> brushRenderData;
        [NoAlias] public NativeList<BlobAssetReference<ChiselBrushRenderBuffer>> patchedRenderBuffers;
        [NoAlias] public HotspotCutState hotspotCut;
        [NoAlias] public FindState       hotspotFind;
        [NoAlias] public WeldMemory      weldMemory;
        [NoAlias, ReadOnly] public NativeList<float3x2> collapseMoves;
        public bool verify, report;

        public void Execute()
        {
            // each brush's own buffer, before the weld's copies: told the hotspot group each surface was mapped with
            hotspotCut.kept.Clear();
            for (int i = 0; i < brushRenderData.Length; i++)
                hotspotCut.kept.Add(brushRenderData[i].brushRenderBuffer);
            OutputModelWeld.FinishIncremental(brushRenderData, weldCopies, ref weldMemory, verify, report, collapseMoves);
            HotspotGroups.StartKinds(brushRenderData, ref hotspotFind);
        }
    }

    // Each brush's surfaces' part in the hotspot groups (HotspotGroups.KindsOf), brush by brush
    [BurstCompile(CompileSynchronously = true)]
    struct SurfaceKindsJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData>         brushRenderData;
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<int>               kindStart;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<byte>  kinds;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<uint4> hashes;

        public void Execute(int b)
        {
            HotspotGroups.KindsOf(b, brushRenderData, hotspotLayouts, kindStart, kinds, hashes);
        }
    }

    // Each group of hotspot surfaces matched on its own
    [BurstCompile(CompileSynchronously = true)]
    struct MatchHotspotGroupsJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData>         brushRenderData;
        [NoAlias, ReadOnly] public NativeArray<HotspotTileData>   hotspotTiles;
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<float4>            hotspotTileSides;
        [NoAlias, ReadOnly] public UnsafeParallelMultiHashMap<int2, HotspotBorderCorner> bordersOf;
        [NoAlias, ReadOnly] public NativeList<int2>     candidates;
        [NoAlias, ReadOnly] public NativeList<float4x4> unfold;
        [NoAlias, ReadOnly] public NativeList<int>      memberStart;
        [NoAlias, ReadOnly] public NativeList<int>      memberOf;
        [NoAlias, ReadOnly] public NativeList<int>      leads;
        [NoAlias, ReadOnly] public NativeReference<uint4> inputs;
        // what each group was matched with before, kept with the tree
        [NoAlias, ReadOnly] public NativeParallelHashMap<uint4, GroupMatchEntry> hotspotMatchCache;

        [NoAlias] public NativeArray<GroupMatch> matches;

        public void Execute(int index)
        {
            var state = new FindState { candidates = candidates, unfold = unfold, memberStart = memberStart, memberOf = memberOf, leads = leads };
            matches[index] = HotspotGroups.Match(leads[index], brushRenderData, hotspotTiles, hotspotLayouts, hotspotTileSides, bordersOf, state,
                                                 hotspotMatchCache, inputs.Value);
        }
    }

    // The groups put together in the model's order, then the hotspot cut planned (CutHotspotBrushesJob)
    [BurstCompile(CompileSynchronously = true)]
    struct PlanHotspotCutJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<float4>            hotspotTileSides;
        [NoAlias, ReadOnly] public NativeArray<ulong>             hotspotVariants;

        // Write
        [NoAlias] public NativeList<BrushData> brushRenderData;
        // what the hotspot cut made of each brush, kept with the tree
        [NoAlias] public NativeParallelHashMap<CompactNodeID, HotspotCutEntry> hotspotCutCache;
        // what each group was matched with, kept with the tree
        [NoAlias] public NativeParallelHashMap<uint4, GroupMatchEntry> hotspotMatchCache;
        // the model's hotspot groups and the brushes left to cut (CutHotspotBrushesJob)
        [NoAlias] public HotspotCutState hotspotCut;
        [NoAlias] public FindState       hotspotFind;

        public void Execute()
        {
            HotspotGroups.Assemble(ref hotspotFind, ref hotspotCut.GroupOf, ref hotspotCut.UnfoldOf, hotspotCut.groups, hotspotCut.lines, hotspotCut.halfPlanes, hotspotCut.cells,
                                   hotspotMatchCache);
            // Then what is drawn of hotspot surfaces is cut into its sections; the collider stays as welded (each brush's key
            // in HotspotCutKeysJob, the brushes to cut in FinishHotspotCutKeysJob)
            OutputHotspotSections.StartCutKeys(brushRenderData, hotspotLayouts, hotspotTileSides, hotspotCut.kept.AsArray(), hotspotVariants, hotspotCutCache.IsCreated, ref hotspotCut);
        }
    }

    // What the hotspot cut reads of each brush (OutputHotspotSections.BrushCutKey), brush by brush
    [BurstCompile(CompileSynchronously = true)]
    struct HotspotCutKeysJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData> brushRenderData;
        [NoAlias, ReadOnly] public NativeReference<UnsafeHashMap<int2, int>>      groupOf;
        [NoAlias, ReadOnly] public NativeReference<UnsafeHashMap<int2, float4x4>> unfoldOf;
        [NoAlias, ReadOnly] public UnsafeParallelMultiHashMap<int2, HotspotBorderCorner> bordersOf;
        [NoAlias, ReadOnly] public NativeArray<uint4> groupKeys;
        [NoAlias, ReadOnly] public NativeReference<uint4> globalKey;
        public bool cached;

        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<uint4> brushKeys;
        [NativeDisableParallelForRestriction] [NoAlias] public NativeArray<byte>  brushCuts;

        public void Execute(int b)
        {
            brushCuts[b] = OutputHotspotSections.BrushCutKey(b, brushRenderData, groupOf.Value, unfoldOf.Value, bordersOf, groupKeys, globalKey.Value, cached, out var key);
            brushKeys[b] = key;
        }
    }

    // The brushes left to cut, and the cache's cuts handed to the others (OutputHotspotSections.FinishCutKeys)
    [BurstCompile(CompileSynchronously = true)]
    struct FinishHotspotCutKeysJob : IJob
    {
        [NoAlias] public NativeList<BrushData> brushRenderData;
        [NoAlias] public NativeParallelHashMap<CompactNodeID, HotspotCutEntry> hotspotCutCache;
        [NoAlias] public HotspotCutState hotspotCut;

        public void Execute()
        {
            OutputHotspotSections.FinishCutKeys(brushRenderData, hotspotCutCache, ref hotspotCut);
        }
    }

    // Each brush FindBrushRenderBuffersJob left to cut, cut on its own
    [BurstCompile(CompileSynchronously = true)]
    struct CutHotspotBrushesJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData>         brushRenderData;
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<float4>            hotspotTileSides;
        [NoAlias, ReadOnly] public NativeArray<ulong>             hotspotVariants;
        [NoAlias, ReadOnly] public NativeReference<UnsafeHashMap<int2, int>>      groupOf;
        [NoAlias, ReadOnly] public NativeReference<UnsafeHashMap<int2, float4x4>> unfoldOf;
        [NoAlias, ReadOnly] public UnsafeParallelMultiHashMap<int2, HotspotBorderCorner> bordersOf;
        [NoAlias, ReadOnly] public NativeList<HotspotGroups.Group>      groups;
        [NoAlias, ReadOnly] public NativeList<HotspotSectionLine>      lines;
        [NoAlias, ReadOnly] public NativeList<HotspotSectionHalfPlane> halfPlanes;
        [NoAlias, ReadOnly] public NativeList<HotspotSectionCell>      cells;

        [NoAlias] public NativeArray<HotspotCutWork> work;

        public void Execute(int index)
        {
            work[index] = OutputHotspotSections.CutPlanned(work[index], brushRenderData, groupOf.Value, unfoldOf.Value, bordersOf, groups, lines, halfPlanes, cells,
                                                           hotspotTileSides, hotspotLayouts, hotspotVariants);
        }
    }

    // The cuts handed to the brushes and the cache
    [BurstCompile(CompileSynchronously = true)]
    struct FinishHotspotCutJob : IJob
    {
        [NoAlias, ReadOnly] public NativeList<HotspotCutWork> work;

        [NoAlias] public NativeList<BrushData> brushRenderData;
        [NoAlias] public NativeList<BlobAssetReference<ChiselBrushRenderBuffer>> patchedRenderBuffers;
        [NoAlias] public NativeParallelHashMap<CompactNodeID, HotspotCutEntry> hotspotCutCache;

        public void Execute()
        {
            OutputHotspotSections.FinishCut(brushRenderData, patchedRenderBuffers, work, hotspotCutCache);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    struct PrepareSubSectionsJob : IJobParallelFor
    {
        // Read
        [NoAlias, ReadOnly] public NativeArray<MeshQuery>   meshQueries;
        [NoAlias, ReadOnly] public NativeList<BrushData>    brushRenderData;

        // Read, Write
        public Allocator allocator;
        [NativeDisableParallelForRestriction]
        [NoAlias] public NativeArray<UnsafeList<SubMeshSurface>> subMeshSurfaces;

        public void Execute(int t)
        {
            int requiredSurfaceCount = 0;
            for (int b = 0, count_b = brushRenderData.Length; b < count_b; b++)
                requiredSurfaceCount += brushRenderData[b].brushSurfaceCount;

            // THIS IS THE SLOWDOWN
            // TODO: store surface separately from brushes, *somehow* make lifetime work
            //              => multiple arrays, one for each meshQuery!
            //              => sorted by surface.layerParameters[meshQuery.layerParameterIndex]!
            //              => this whole job could be removed
            //
            // TODO: store surface info and its vertices/indices separately, both sequentially in arrays
            // TODO: store surface vertices/indices sequentially in a big array, *somehow* make ordering work

            if (subMeshSurfaces[t].IsCreated)
            {
                // should not happen
                subMeshSurfaces[t].Dispose();
                subMeshSurfaces[t] = default;
            }

            var subMeshSurfaceList = new UnsafeList<SubMeshSurface>(math.max(1, requiredSurfaceCount), allocator);

            // A collider is a mesh per physics material and chunk, so a change rebakes only the chunks it touched
            var chunked = meshQueries[t].LayerParameterIndex == SurfaceParameterIndex.PhysicsMaterial;
            var brushChunks = new NativeArray<uint>(math.max(1, brushRenderData.Length), Allocator.Temp);
            if (chunked)
            {
                for (int b = 0, count_b = brushRenderData.Length; b < count_b; b++)
                    brushChunks[b] = ChunkOf(ref brushRenderData[b].brushRenderBuffer.Value);
            }

            // One key per surface of this query. Sorted, this both yields the distinct parameters and
            // counts each group's surfaces, so the per-brush blobs are walked twice, not three times.
            var parameters = new NativeList<SurfaceGroupKey>(math.max(1, requiredSurfaceCount), Allocator.Temp);
            for (int b = 0, count_b = brushRenderData.Length; b < count_b; b++)
            {
                ref var surfaces = ref brushRenderData[b].brushRenderBuffer.Value.querySurfaces[t].surfaces;
                for (int s = 0; s < surfaces.Length; s++)
                    parameters.AddNoResize(new SurfaceGroupKey { parameter = surfaces[s].surfaceParameter, chunk = brushChunks[b] });
            }
            var surfaceCount = parameters.Length;
            subMeshSurfaceList.Resize(surfaceCount, NativeArrayOptions.UninitializedMemory);
            if (surfaceCount == 0)
            {
                parameters.Dispose();
                brushChunks.Dispose();
                subMeshSurfaces[t] = subMeshSurfaceList;
                return;
            }

            parameters.Sort();

            // Run-length encode the sorted keys into distinct parameters + the offset of each group
            var groupOffsets = new NativeList<int>(surfaceCount + 1, Allocator.Temp);
            int uniqueCount = 0;
            for (int i = 0; i < surfaceCount; i++)
            {
                if (uniqueCount == 0 || !parameters[uniqueCount - 1].Equals(parameters[i]))
                {
                    parameters[uniqueCount++] = parameters[i];
                    groupOffsets.AddNoResize(i);
                }
            }
            parameters.Length = uniqueCount;

            // Scatter, keeping brush order within a group
            for (int b = 0, count_b = brushRenderData.Length; b < count_b; b++)
            {
                var brushData         = brushRenderData[b];
                var brushRenderBuffer = brushData.brushRenderBuffer;
                ref var querySurfaces = ref brushRenderBuffer.Value.querySurfaces[t];
                ref var brushNodeID   = ref querySurfaces.brushNodeID;
				ref var surfaces      = ref querySurfaces.surfaces;

                for (int s = 0; s < surfaces.Length; s++)
                {
                    var group = FindGroup(parameters, new SurfaceGroupKey { parameter = surfaces[s].surfaceParameter, chunk = brushChunks[b] });
                    var index = groupOffsets[group];
                    groupOffsets[group] = index + 1;
                    subMeshSurfaceList[index] = new SubMeshSurface
                    {
                        brushNodeID       = brushNodeID,
                        surfaceIndex      = surfaces[s].surfaceIndex,
                        surfaceParameter  = surfaces[s].surfaceParameter,
                        chunk             = brushChunks[b],
                        vertexCount       = surfaces[s].vertexCount,
                        indexCount        = surfaces[s].indexCount,
                        surfaceHashValue  = surfaces[s].surfaceHashValue,
                        geometryHashValue = surfaces[s].geometryHashValue,
                        brushRenderBuffer = brushRenderBuffer,
                    };
                }
            }
            groupOffsets.Dispose();
            parameters.Dispose();
            brushChunks.Dispose();

            subMeshSurfaces[t] = subMeshSurfaceList;
        }

        const float kChunkSize = 16.0f;

        // The cell of space a brush's output is centred in
        static uint ChunkOf(ref ChiselBrushRenderBuffer brushRenderBuffer)
        {
            ref var surfaces = ref brushRenderBuffer.surfaces;
            var min = new float3(float.PositiveInfinity);
            var max = new float3(float.NegativeInfinity);
            for (int s = 0; s < surfaces.Length; s++)
            {
                if (surfaces[s].indexCount == 0)
                    continue;
                min = math.min(min, surfaces[s].aabb.Min);
                max = math.max(max, surfaces[s].aabb.Max);
            }
            if (math.any(min > max))
                return 0;
            var cell = (int3)math.floor((min + max) * (0.5f / kChunkSize));
            return math.hash(cell);
        }

        // Index of `value` in the sorted, distinct `parameters` (every value looked up came from it)
        static int FindGroup(in NativeList<SurfaceGroupKey> parameters, SurfaceGroupKey value)
        {
            int lo = 0, hi = parameters.Length - 1;
            while (lo < hi)
            {
                var mid = (lo + hi) >> 1;
                if (parameters[mid].CompareTo(value) < 0) lo = mid + 1; else hi = mid;
            }
            return lo;
        }
    }
    
    [BurstCompile(CompileSynchronously = true)]
    struct SortSurfacesParallelJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public NativeArray<MeshQuery>                  meshQueries;
        [NoAlias, ReadOnly] public NativeArray<UnsafeList<SubMeshSurface>> subMeshSurfaces;

        // Write
        [NoAlias, WriteOnly] public NativeList<SubMeshDescriptions> subMeshDescriptions;

        public void Execute()
        {
            // TODO: figure out why this order matters
            for (int t = 0; t < subMeshSurfaces.Length; t++)
            {
                Execute(t);
            }
        }

        public void Execute(int t)
        {
            var subMeshSurfaceList  = subMeshSurfaces[t];
            var surfaceCount        = subMeshSurfaceList.Length;
            if (surfaceCount == 0)
                return;

            var meshQuery       = meshQueries[t];

			NativeList<SubMeshDescriptions> sectionSubMeshCounts; 
			using var _sectionSubMeshCounts = sectionSubMeshCounts = new NativeList<SubMeshDescriptions>(surfaceCount, Allocator.Temp);
            var currentSubMesh  = new SubMeshDescriptions
            {
                meshQueryIndex      = t,
                subMeshQueryIndex   = 0,
                meshQuery           = meshQuery,
                surfacesOffset      = 0,
                surfaceParameter    = subMeshSurfaceList[0].surfaceParameter,
                chunk               = subMeshSurfaceList[0].chunk
            };
            for (int b = 0; b < surfaceCount; b++)
            {
                var subMeshSurface              = subMeshSurfaceList[b];
                var surfaceParameter            = subMeshSurface.surfaceParameter;
                var surfaceVertexCount          = subMeshSurface.vertexCount;
                var surfaceIndexCount           = subMeshSurface.indexCount;

                if (currentSubMesh.surfaceParameter != surfaceParameter || currentSubMesh.chunk != subMeshSurface.chunk)
                {
                    // Store the previous subMeshCount
                    if (currentSubMesh.indexCount > 0 && currentSubMesh.vertexCount > 0)
                        sectionSubMeshCounts.AddNoResize(currentSubMesh);
                        
                    // Create the new SubMeshCount
                    currentSubMesh.surfaceHashValue = 0;
                    currentSubMesh.geometryHashValue = 0;
                    
                    currentSubMesh.indexCount = 0;
                    currentSubMesh.vertexCount = 0;

					currentSubMesh.surfacesOffset += currentSubMesh.surfacesCount;
					currentSubMesh.surfacesCount = 0;
                    currentSubMesh.surfaceParameter = surfaceParameter;
                    currentSubMesh.chunk = subMeshSurface.chunk;
                    currentSubMesh.subMeshQueryIndex++;
                } 

                currentSubMesh.indexCount += surfaceIndexCount;
                currentSubMesh.vertexCount += surfaceVertexCount;
                currentSubMesh.surfaceHashValue  = math.hash(new uint2(currentSubMesh.surfaceHashValue,  subMeshSurface.surfaceHashValue));
                currentSubMesh.geometryHashValue = math.hash(new uint2(currentSubMesh.geometryHashValue, subMeshSurface.geometryHashValue));
                currentSubMesh.surfacesCount++;
            }

            // Store the last subMeshCount
            if (currentSubMesh.indexCount > 0 && currentSubMesh.vertexCount > 0)
                sectionSubMeshCounts.AddNoResize(currentSubMesh);

            subMeshDescriptions.AddRangeNoResize(sectionSubMeshCounts);
		}
    }
    
    [BurstCompile(CompileSynchronously = true)]
    struct GatherSurfacesJob : IJob
    {
        // Read / Write
        [NoAlias] public NativeList<SubMeshDescriptions>              subMeshDescriptions;

        // Write
        [NoAlias, WriteOnly] public NativeList<SubMeshSection>  subMeshSections;
            
        // A query whose materials each get a mesh of their own: one that draws (a shadow-only one draws everything with
        // one material)
        internal static bool OneMeshPerMaterial(MeshQuery query)
        {
            return query.LayerParameterIndex == SurfaceParameterIndex.RenderMaterial &&
                   (query.LayerQuery & SurfaceDestinationFlags.Renderable) != 0;
        }

        public void Execute()
        {
            if (subMeshDescriptions.Length == 0)
                return;

            int descriptionIndex = 0;
            //var contentsIndex = 0;
            if (subMeshDescriptions[0].meshQuery.LayerParameterIndex == SurfaceParameterIndex.None ||
                subMeshDescriptions[0].meshQuery.LayerParameterIndex == SurfaceParameterIndex.RenderMaterial)
            {
                var prevQuery = subMeshDescriptions[0].meshQuery;
                var startIndex = 0;
                for (; descriptionIndex < subMeshDescriptions.Length; descriptionIndex++)
                {
                    var subMeshCount = subMeshDescriptions[descriptionIndex];
                    // Exit when layerParameterIndex is no longer LayerParameter1/None
                    if (subMeshCount.meshQuery.LayerParameterIndex != SurfaceParameterIndex.None &&
                        subMeshCount.meshQuery.LayerParameterIndex != SurfaceParameterIndex.RenderMaterial)
                        break;

                    // a render query's materials each get a mesh of their own, so a mesh changes only with its material
                    var currQuery = subMeshCount.meshQuery;
                    if (descriptionIndex == startIndex ||
                        (prevQuery == currQuery && !OneMeshPerMaterial(currQuery)))
                    {
                        continue;
                    }

                    int totalVertexCount = 0;
                    int totalIndexCount = 0;
                    for (int i = startIndex; i < descriptionIndex; i++)
                    {
                        totalVertexCount += subMeshDescriptions[i].vertexCount;
                        totalIndexCount += subMeshDescriptions[i].indexCount;
                    }

                    // Group by all subMeshCounts with same query
                    subMeshSections.AddNoResize(new SubMeshSection
                    {
                        meshQuery           = subMeshDescriptions[startIndex].meshQuery,
                        startIndex          = startIndex, 
                        endIndex            = descriptionIndex,
                        totalVertexCount    = totalVertexCount,
                        totalIndexCount     = totalIndexCount,
                    });

                    startIndex = descriptionIndex;
                    prevQuery = currQuery;
                }

                {
                    int totalVertexCount = 0;
                    int totalIndexCount = 0;
                    for (int i = startIndex; i < descriptionIndex; i++)
                    {
                        totalVertexCount += subMeshDescriptions[i].vertexCount;
                        totalIndexCount  += subMeshDescriptions[i].indexCount;
                    }

                    subMeshSections.AddNoResize(new SubMeshSection
                    {
                        meshQuery           = subMeshDescriptions[startIndex].meshQuery,
                        startIndex          = startIndex,
                        endIndex            = descriptionIndex,
                        totalVertexCount    = totalVertexCount,
                        totalIndexCount     = totalIndexCount
                    });
                }
            }
                

            if (descriptionIndex < subMeshDescriptions.Length &&
                subMeshDescriptions[descriptionIndex].meshQuery.LayerParameterIndex == SurfaceParameterIndex.PhysicsMaterial)
            {
                Debug.Assert(subMeshDescriptions[subMeshDescriptions.Length - 1].meshQuery.LayerParameterIndex == SurfaceParameterIndex.PhysicsMaterial);

                // Loop through all subMeshCounts with LayerParameter2, and create collider meshes from them
                for (int i = 0; descriptionIndex < subMeshDescriptions.Length; descriptionIndex++, i++)
                {
                    var subMeshCount = subMeshDescriptions[descriptionIndex];

                    // Exit when layerParameterIndex is no longer LayerParameter2
                    if (subMeshCount.meshQuery.LayerParameterIndex != SurfaceParameterIndex.PhysicsMaterial)
                        break;

                    subMeshSections.AddNoResize(new SubMeshSection
                    {
                        meshQuery           = subMeshCount.meshQuery,
                        startIndex          = descriptionIndex,
                        endIndex            = descriptionIndex,
                        totalVertexCount    = subMeshCount.vertexCount,
                        totalIndexCount     = subMeshCount.indexCount
                    });
                }
            }
        }
    }
    
    [BurstCompile(CompileSynchronously = true)]
    struct GenerateMeshDescriptionJob : IJob
    {
        [NoAlias, ReadOnly] public NativeList<SubMeshDescriptions> subMeshDescriptions;

        [NativeDisableParallelForRestriction]
        [NoAlias] public NativeList<GeneratedMeshDescription> meshDescriptions;

        public void Execute()
        {
            if (meshDescriptions.Capacity < subMeshDescriptions.Length)
                meshDescriptions.Capacity = subMeshDescriptions.Length;

            for (int i = 0; i < subMeshDescriptions.Length; i++)
            {
                var subMesh = subMeshDescriptions[i];

                var description = new GeneratedMeshDescription
                {
                    meshQuery           = subMesh.meshQuery,
                    surfaceParameter    = subMesh.surfaceParameter,
                    chunk               = subMesh.chunk,
                    meshQueryIndex      = subMesh.meshQueryIndex,
                    subMeshQueryIndex   = subMesh.subMeshQueryIndex,

                    geometryHashValue   = subMesh.geometryHashValue,
                    surfaceHashValue    = subMesh.surfaceHashValue,

                    vertexCount         = subMesh.vertexCount,
                    indexCount          = subMesh.indexCount
                };

                meshDescriptions.Add(description);
            }
        }
    }    
}
