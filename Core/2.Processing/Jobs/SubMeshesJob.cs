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

		public int vertexCount;
		public int indexCount;

		public uint geometryHashValue;
		public uint surfaceHashValue;
		public BlobAssetReference<ChiselBrushRenderBuffer> brushRenderBuffer;
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

        // Write
        [NoAlias] public NativeList<BrushData> brushRenderData;

        [NoAlias] public NativeReference<int> surfaceCountRef;
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
            // (the brushes it changed are stored again by CopyWeldedBrushesJob)
            OutputModelWeld.PlanWeld(brushRenderData, ref weld, out _, out _, out _, out _, collapseMoves);
        }
    }

    // Each brush the weld changed, stored again on its own
    [BurstCompile(CompileSynchronously = true)]
    struct CopyWeldedBrushesJob : IJobParallelForDefer
    {
        [NoAlias, ReadOnly] public NativeArray<BrushData>              brushRenderData;
        [NoAlias, ReadOnly] public NativeList<OutputModelWeld.Working> work;
        [NoAlias, ReadOnly] public NativeParallelHashMap<long, int>    workIndex;

        [NoAlias] public NativeArray<WeldCopy> copies;

        public void Execute(int index)
        {
            copies[index] = OutputModelWeld.CopyPlanned(copies[index], brushRenderData, work, workIndex);
        }
    }

    // The weld's copies handed to their brushes, then who is grouped with whom (MatchHotspotGroupsJob)
    [BurstCompile(CompileSynchronously = true)]
    struct PrepareHotspotGroupsJob : IJob
    {
        // Read
        [NoAlias, ReadOnly] public NativeList<WeldCopy>           weldCopies;
        [NoAlias, ReadOnly] public NativeArray<HotspotTileData>   hotspotTiles;
        [NoAlias, ReadOnly] public NativeArray<HotspotLayoutData> hotspotLayouts;
        [NoAlias, ReadOnly] public NativeArray<HotspotLink>       hotspotLinks;
        [NoAlias, ReadOnly] public NativeArray<float4>            hotspotTileSides;

        // Write
        [NoAlias] public NativeList<BrushData> brushRenderData;
        [NoAlias] public NativeList<BlobAssetReference<ChiselBrushRenderBuffer>> patchedRenderBuffers;
        [NoAlias] public HotspotCutState hotspotCut;
        [NoAlias] public FindState       hotspotFind;

        public void Execute()
        {
            // each brush's own buffer, before the weld's copies: told the hotspot group each surface was mapped with
            hotspotCut.kept.Clear();
            for (int i = 0; i < brushRenderData.Length; i++)
                hotspotCut.kept.Add(brushRenderData[i].brushRenderBuffer);
            OutputModelWeld.FinishWeld(brushRenderData, patchedRenderBuffers, weldCopies);
            HotspotGroups.Borders(brushRenderData, hotspotLinks, ref hotspotCut.bordersOf);
            HotspotGroups.Prepare(brushRenderData, hotspotTiles, hotspotLayouts, hotspotLinks, ref hotspotFind);
            hotspotFind.inputs.Value = HotspotGroups.InputsKey(hotspotTiles, hotspotLayouts, hotspotTileSides);
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
            // Then what is drawn of hotspot surfaces is cut into its sections; the collider stays as welded
            OutputHotspotSections.PlanCutAfterGroups(brushRenderData, hotspotLayouts, hotspotTileSides, hotspotCut.kept.AsArray(), hotspotVariants, hotspotCutCache, ref hotspotCut);
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

            // One key per surface of this query. Sorted, this both yields the distinct parameters and
            // counts each group's surfaces, so the per-brush blobs are walked twice, not three times.
            var parameters = new NativeList<ulong>(math.max(1, requiredSurfaceCount), Allocator.Temp);
            for (int b = 0, count_b = brushRenderData.Length; b < count_b; b++)
            {
                ref var surfaces = ref brushRenderData[b].brushRenderBuffer.Value.querySurfaces[t].surfaces;
                for (int s = 0; s < surfaces.Length; s++)
                    parameters.AddNoResize(surfaces[s].surfaceParameter);
            }
            var surfaceCount = parameters.Length;
            subMeshSurfaceList.Resize(surfaceCount, NativeArrayOptions.UninitializedMemory);
            if (surfaceCount == 0)
            {
                parameters.Dispose();
                subMeshSurfaces[t] = subMeshSurfaceList;
                return;
            }

            parameters.Sort();

            // Run-length encode the sorted keys into distinct parameters + the offset of each group
            var groupOffsets = new NativeList<int>(surfaceCount + 1, Allocator.Temp);
            int uniqueCount = 0;
            for (int i = 0; i < surfaceCount; i++)
            {
                if (uniqueCount == 0 || parameters[uniqueCount - 1] != parameters[i])
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
                    var group = FindGroup(parameters, surfaces[s].surfaceParameter);
                    var index = groupOffsets[group];
                    groupOffsets[group] = index + 1;
                    subMeshSurfaceList[index] = new SubMeshSurface
                    {
                        brushNodeID       = brushNodeID,
                        surfaceIndex      = surfaces[s].surfaceIndex,
                        surfaceParameter  = surfaces[s].surfaceParameter,
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

            subMeshSurfaces[t] = subMeshSurfaceList;
        }

        // Index of `value` in the sorted, distinct `parameters` (every value looked up came from it)
        static int FindGroup(in NativeList<ulong> parameters, ulong value)
        {
            int lo = 0, hi = parameters.Length - 1;
            while (lo < hi)
            {
                var mid = (lo + hi) >> 1;
                if (parameters[mid] < value) lo = mid + 1; else hi = mid;
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
                surfaceParameter    = subMeshSurfaceList[0].surfaceParameter
            };
            for (int b = 0; b < surfaceCount; b++)
            {
                var subMeshSurface              = subMeshSurfaceList[b];
                var surfaceParameter            = subMeshSurface.surfaceParameter;
                var surfaceVertexCount          = subMeshSurface.vertexCount;
                var surfaceIndexCount           = subMeshSurface.indexCount;

                if (currentSubMesh.surfaceParameter != surfaceParameter)
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

                    var currQuery = subMeshCount.meshQuery;
                    if (prevQuery == currQuery)
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
