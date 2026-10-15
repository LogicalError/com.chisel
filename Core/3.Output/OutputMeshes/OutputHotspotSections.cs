using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;

namespace Chisel.Core
{
    // After the model's weld: the surfaces with hotspot sections are cut where their sections meet and given their uvs.
    // Only what is drawn is cut; the collider keeps the welded triangles (ChiselSurfaceRenderBuffer.physicsIndices).
    // A brush's cut surfaces, with what they were made from
    internal struct HotspotCutEntry
    {
        public uint4 key;
        public BlobAssetReference<ChiselBrushRenderBuffer> result;
        public bool used;
    }

    internal static class OutputHotspotSections
    {
        // kept: each brush's own buffer, as the tree keeps it (before the weld patched any); it is told the group each of its
        // surfaces was mapped with, which the editor reads (the patched copies are gone after the meshes are made)
        public static void CutModel(NativeList<BrushData> brushes, NativeList<BlobAssetReference<ChiselBrushRenderBuffer>> patched,
                                    NativeArray<HotspotTileData> tiles, NativeArray<HotspotLayoutData> layouts, NativeArray<HotspotLink> links,
                                    NativeArray<float4> tileSides = default, NativeArray<BlobAssetReference<ChiselBrushRenderBuffer>> kept = default,
                                    NativeArray<ulong> variants = default, NativeParallelHashMap<CompactNodeID, HotspotCutEntry> cache = default)
        {
            // coplanar neighbours, of any brush, mapped as one surface (HotspotGroups)
            var groupOf    = new UnsafeHashMap<int2, int>(64, Allocator.Temp);
            var unfoldOf   = new UnsafeHashMap<int2, float4x4>(8, Allocator.Temp);
            var groups     = new NativeList<HotspotGroups.Group>(8, Allocator.Temp);
            var lines      = new NativeList<HotspotSectionLine>(32, Allocator.Temp);
            var halfPlanes = new NativeList<HotspotSectionHalfPlane>(64, Allocator.Temp);
            var cells      = new NativeList<HotspotSectionCell>(32, Allocator.Temp);
            // and the edges given a border or none by hand
            var bordersOf  = new UnsafeParallelMultiHashMap<int2, HotspotBorderCorner>(8, Allocator.Temp);
            HotspotGroups.Borders(brushes, links, ref bordersOf);
            HotspotGroups.Find(brushes, tiles, layouts, links, bordersOf, ref groupOf, ref unfoldOf, groups, lines, halfPlanes, cells, tileSides);
            for (int b = 0; kept.IsCreated && b < kept.Length && b < brushes.Length; b++)
            {
                if (!kept[b].IsCreated)
                    continue;
                ref var surfaces = ref kept[b].Value.surfaces;
                for (int s = 0; s < surfaces.Length; s++)
                {
                    var grouped = groupOf.TryGetValue(new int2(b, s), out var g);
                    surfaces[s].hotspot.group           = grouped ? g + 1 : 0;
                    surfaces[s].hotspot.groupVariations = grouped ? groups[g].info.variations : 0;
                }
            }
            var cached = cache.IsCreated;
            var groupKeys = new NativeArray<uint4>(cached ? groups.Length : 0, Allocator.Temp);
            var globalKey = default(uint4);
            if (cached)
            {
                for (int g = 0; g < groups.Length; g++)
                    groupKeys[g] = HashGroup(groups[g], lines, halfPlanes, cells);
                globalKey = HashInputs(layouts, tileSides, variants);
            }
            using var rules = new NativeList<HotspotBorderCorner>(4, Allocator.Temp);
            for (int b = 0; b < brushes.Length; b++)
            {
                var nodeID = brushes[b].brushIndexOrder.compactNodeID;
                if (!brushes[b].brushRenderBuffer.IsCreated ||
                    (!HasSections(ref brushes[b].brushRenderBuffer.Value) && !InGroup(ref brushes[b].brushRenderBuffer.Value, b, groupOf)))
                {
                    if (cached && cache.TryGetValue(nodeID, out var stale))
                    {
                        if (stale.result.IsCreated)
                            stale.result.Dispose();
                        cache.Remove(nodeID);
                    }
                    continue;
                }
                // what the cut reads: the brush as the weld left it, its groups, the borders given by hand, the layouts
                var key = default(uint4);
                var keyed = false;
                if (cached)
                    keyed = BrushKey(brushes[b].brushRenderBuffer, b, groupOf, unfoldOf, bordersOf, groupKeys, globalKey, rules, out key);
                BlobAssetReference<ChiselBrushRenderBuffer> copy;
                if (keyed && cache.TryGetValue(nodeID, out var entry) && entry.result.IsCreated && entry.key.Equals(key))
                {
                    copy = entry.result;
                    entry.used = true;
                    cache[nodeID] = entry;
                }
                else
                {
                    copy = Cut(brushes[b].brushRenderBuffer, b, groupOf, unfoldOf, bordersOf, groups, lines, halfPlanes, cells, tileSides, layouts, variants);
                    if (keyed)
                    {
                        if (cache.TryGetValue(nodeID, out var old) && old.result.IsCreated)
                            old.result.Dispose();
                        cache[nodeID] = new HotspotCutEntry { key = key, result = copy, used = true };
                    }
                    else
                        patched.Add(copy);
                }
                var data = brushes[b];
                data.brushRenderBuffer = copy;
                brushes[b] = data;
            }
            if (cached)
            {
                // the brushes this update did not cut are gone, or are not cut any more
                using var gone = new NativeList<CompactNodeID>(Allocator.Temp);
                foreach (var pair in cache)
                {
                    if (!pair.Value.used)
                        gone.Add(pair.Key);
                }
                foreach (var id in gone)
                {
                    if (cache.TryGetValue(id, out var dead) && dead.result.IsCreated)
                        dead.result.Dispose();
                    cache.Remove(id);
                }
                using var left = cache.GetKeyArray(Allocator.Temp);
                foreach (var id in left)
                {
                    var kept2 = cache[id];
                    kept2.used = false;
                    cache[id] = kept2;
                }
            }
            groupOf.Dispose(); unfoldOf.Dispose(); bordersOf.Dispose(); groups.Dispose(); lines.Dispose(); halfPlanes.Dispose(); cells.Dispose();
        }

        // The size of a blob's data: it sits in the header in front of it (Unity.Entities BlobAssetHeader, 32 bytes, Length at 8)
        static unsafe bool BlobBytes(BlobAssetReference<ChiselBrushRenderBuffer> blob, out byte* data, out int length)
        {
            data = (byte*)blob.GetUnsafePtr();
            length = *(int*)(data - 32 + 8);
            return length > 0 && length < (1 << 30);
        }

        static unsafe uint4 Hash(void* data, long length) { return length == 0 ? default : xxHash3.Hash128(data, length); }

        // The group as the cut reads it: where its sections start in the model's flat lists changes with the other groups
        static unsafe uint4 HashGroup(HotspotGroups.Group group, NativeList<HotspotSectionLine> lines, NativeList<HotspotSectionHalfPlane> halfPlanes,
                                      NativeList<HotspotSectionCell> cells)
        {
            var start = group.sectionStart;
            var count = group.sectionCount;
            group.sectionStart = int3.zero;
            var parts = stackalloc uint4[4];
            parts[0] = Hash(&group, UnsafeUtility.SizeOf<HotspotGroups.Group>());
            parts[1] = Hash((byte*)lines.GetUnsafePtr()      + (long)start.x * UnsafeUtility.SizeOf<HotspotSectionLine>(),      (long)count.x * UnsafeUtility.SizeOf<HotspotSectionLine>());
            parts[2] = Hash((byte*)halfPlanes.GetUnsafePtr() + (long)start.y * UnsafeUtility.SizeOf<HotspotSectionHalfPlane>(), (long)count.y * UnsafeUtility.SizeOf<HotspotSectionHalfPlane>());
            parts[3] = Hash((byte*)cells.GetUnsafePtr()      + (long)start.z * UnsafeUtility.SizeOf<HotspotSectionCell>(),      (long)count.z * UnsafeUtility.SizeOf<HotspotSectionCell>());
            return Hash(parts, 4 * 16);
        }

        static unsafe uint4 HashInputs(NativeArray<HotspotLayoutData> layouts, NativeArray<float4> tileSides, NativeArray<ulong> variants)
        {
            var parts = stackalloc uint4[3];
            parts[0] = layouts.IsCreated   ? Hash(layouts.GetUnsafeReadOnlyPtr(),   (long)layouts.Length   * UnsafeUtility.SizeOf<HotspotLayoutData>()) : default;
            parts[1] = tileSides.IsCreated ? Hash(tileSides.GetUnsafeReadOnlyPtr(), (long)tileSides.Length * UnsafeUtility.SizeOf<float4>()) : default;
            parts[2] = variants.IsCreated  ? Hash(variants.GetUnsafeReadOnlyPtr(),  (long)variants.Length  * UnsafeUtility.SizeOf<ulong>()) : default;
            return Hash(parts, 3 * 16);
        }

        // One key for everything Cut reads for the brush; false when the brush cannot be keyed (it is cut every time then)
        static unsafe bool BrushKey(BlobAssetReference<ChiselBrushRenderBuffer> source, int brush, UnsafeHashMap<int2, int> groupOf, UnsafeHashMap<int2, float4x4> unfoldOf,
                                    in UnsafeParallelMultiHashMap<int2, HotspotBorderCorner> bordersOf, NativeArray<uint4> groupKeys, uint4 globalKey,
                                    NativeList<HotspotBorderCorner> rules, out uint4 key)
        {
            key = default;
            if (!BlobBytes(source, out var data, out var length))
                return false;
            var buffer = new UnsafeAppendBuffer(256, 16, Allocator.Temp);
            try
            {
                buffer.Add(Hash(data, length));
                buffer.Add(globalKey);
                ref var surfaces = ref source.Value.surfaces;
                for (int s = 0; s < surfaces.Length; s++)
                {
                    if (surfaces[s].indices.Length > 0 && groupOf.TryGetValue(new int2(brush, s), out var g))
                    {
                        buffer.Add(1);
                        buffer.Add(groupKeys[g]);
                        var unfolded = unfoldOf.TryGetValue(new int2(brush, s), out var unfold);
                        buffer.Add(unfolded);
                        if (unfolded)
                            buffer.Add(unfold);
                        continue;
                    }
                    HotspotGroups.BordersOf(new int2(brush, s), bordersOf, rules);
                    buffer.Add(2 + rules.Length);
                    for (int r = 0; r < rules.Length; r++)
                        buffer.Add(rules[r]);
                }
                key = xxHash3.Hash128(buffer.Ptr, buffer.Length);
                return true;
            }
            finally
            {
                buffer.Dispose();
            }
        }

        static bool InGroup(ref ChiselBrushRenderBuffer buffer, int brush, UnsafeHashMap<int2, int> groupOf)
        {
            for (int s = 0; s < buffer.surfaces.Length; s++)
            {
                if (groupOf.ContainsKey(new int2(brush, s)))
                    return true;
            }
            return false;
        }

        static bool HasSections(ref ChiselBrushRenderBuffer buffer)
        {
            for (int s = 0; s < buffer.surfaces.Length; s++)
            {
                if (buffer.surfaces[s].sectionInput.count > 0 && buffer.surfaces[s].indices.Length > 0)
                    return true;
            }
            return false;
        }

        static BlobAssetReference<ChiselBrushRenderBuffer> Cut(BlobAssetReference<ChiselBrushRenderBuffer> source, int brush, UnsafeHashMap<int2, int> groupOf,
                                                               UnsafeHashMap<int2, float4x4> unfoldOf, in UnsafeParallelMultiHashMap<int2, HotspotBorderCorner> bordersOf,
                                                               NativeList<HotspotGroups.Group> groups, NativeList<HotspotSectionLine> groupLines,
                                                               NativeList<HotspotSectionHalfPlane> groupHalfPlanes, NativeList<HotspotSectionCell> groupCells,
                                                               NativeArray<float4> tileSides, NativeArray<HotspotLayoutData> layouts, NativeArray<ulong> variants)
        {
            var builder = new BlobBuilder(Allocator.Temp);
            try
            {
                ref var original = ref source.Value;
                ref var root     = ref builder.ConstructRoot<ChiselBrushRenderBuffer>();
                var surfaces     = builder.Allocate(ref root.surfaces, original.surfaces.Length);
                for (int s = 0; s < original.surfaces.Length; s++)
                {
                    ref var from = ref original.surfaces[s];
                    ref var to   = ref surfaces[s];
                    to.surfaceIndex          = from.surfaceIndex;
                    to.baseSurfaceIndex      = from.baseSurfaceIndex;
                    to.decalEntityID         = from.decalEntityID;
                    to.destinationFlags      = from.destinationFlags;
                    to.destinationParameters = from.destinationParameters;
                    to.outputFlags           = from.outputFlags;
                    if (from.indices.Length > 0 && groupOf.TryGetValue(new int2(brush, s), out var g))
                    {
                        var unfolded = unfoldOf.TryGetValue(new int2(brush, s), out var unfold);
                        // the group's material, when its layout has more than one
                        var group = groups[g];
                        if (group.sheetMaterial != 0)
                            to.destinationParameters.parameter1 = group.sheetMaterial;
                        else if (group.layout >= 0 && group.layout < layouts.Length)
                        {
                            // on the layout's own tiles: one of its materials, or its own back (a member's own match may
                            // have given it another sheet's)
                            var layout = layouts[group.layout];
                            if (variants.IsCreated && layout.variantCount > 1 && layout.variantStart + group.variant < variants.Length)
                                to.destinationParameters.parameter1 = variants[layout.variantStart + group.variant];
                            else if (layout.material != 0)
                                to.destinationParameters.parameter1 = layout.material;
                        }
                        MapInGroup(builder, ref from, ref to, groups[g], g, unfolded ? unfold : float4x4.identity, unfolded, groupLines, groupHalfPlanes, groupCells);
                        continue;
                    }
                    if (from.sectionInput.count == 0 || from.indices.Length == 0)
                    {
                        to.CopyGeometry(builder, ref from);
                        to.CopyHotspot(builder, ref from);
                        continue;
                    }

                    using var indices  = ToList(ref from.indices);
                    using var collider = ToList(ref from.colliderVertices);
                    using var select   = ToList(ref from.selectVertices);
                    using var render   = ToList(ref from.renderVertices);
                    var input    = from.sectionInput;
                    var budget   = input.MaxTriangles;
                    var sections = HotspotSections.Create(Allocator.Temp);
                    using var rules = new NativeList<HotspotBorderCorner>(4, Allocator.Temp);
                    HotspotGroups.BordersOf(new int2(brush, s), bordersOf, rules);
                    // around what the surface draws (its border, its interior), unless it is a triangle (exact already) or
                    // repeats
                    var built = input.count == 4 && !input.Repeated
                              ? sections.BuildConforming(input, indices, collider, rules.AsArray(), tileSides)
                              : sections.Build(input);
                    if (!built)
                    {
                        sections.Dispose();
                        to.CopyGeometry(builder, ref from);
                        to.CopyHotspot(builder, ref from);
                        continue;
                    }
                    var laid = CutAndMap(sections, input, input.right, input.up, true, indices, collider, select, render);
                    // over its budget: without the interior's repeats; still over: drawn as it was, uncut
                    var limited = indices.Length / 3 > budget;
                    if (limited)
                    {
                        Load(ref from, indices, collider, select, render);
                        laid = CutAndMap(sections, input, input.right, input.up, false, indices, collider, select, render);
                    }
                    sections.Dispose();
                    if (indices.Length / 3 > budget)
                    {
                        to.CopyGeometry(builder, ref from);
                        to.CopyHotspot(builder, ref from);
                        to.hotspot.flags |= HotspotSurfaceFlags.BudgetLimited;
                        continue;
                    }
                    // a surface whose cut went wrong is drawn uncut: one bad point would hide the whole model (its
                    // bounds)
                    if (!Finite(collider, render))
                    {
                        to.CopyGeometry(builder, ref from);
                        to.CopyHotspot(builder, ref from);
                        continue;
                    }
                    to.Store(builder, indices, collider, select, render);
                    // the collider keeps what the weld left, uncut
                    builder.Construct(ref to.physicsIndices,  ref from.indices);
                    builder.Construct(ref to.physicsVertices, ref from.colliderVertices);
                    to.CopySections(builder, ref from);
                    to.hotspot.repeats = laid;
                    if (limited)
                        to.hotspot.flags |= HotspotSurfaceFlags.BudgetLimited;
                }

                // The mesh queries' lists, with the cut surfaces' counts and hashes
                var queries = builder.Allocate(ref root.querySurfaces, original.querySurfaces.Length);
                using var list = new NativeList<ChiselQuerySurface>(Allocator.Temp);
                for (int q = 0; q < original.querySurfaces.Length; q++)
                {
                    ref var from = ref original.querySurfaces[q];
                    list.Clear();
                    for (int e = 0; e < from.surfaces.Length; e++)
                    {
                        var entry = from.surfaces[e];
                        ref var stored = ref surfaces[entry.surfaceIndex];
                        if (stored.vertexCount == 0 || stored.indexCount == 0)
                            continue;
                        // the render material the cut gave it (a group's sheet or variant), where the entry had the old one
                        if (entry.surfaceParameter != 0 && entry.surfaceParameter == original.surfaces[entry.surfaceIndex].destinationParameters.parameter1)
                            entry.surfaceParameter = stored.destinationParameters.parameter1;
                        entry.vertexCount       = stored.vertexCount;
                        entry.indexCount        = stored.indexCount;
                        entry.geometryHashValue = stored.geometryHashValue;
                        entry.surfaceHashValue  = stored.surfaceHashValue;
                        list.Add(entry);
                    }
                    queries[q].brushNodeID = from.brushNodeID;
                    builder.Construct(ref queries[q].surfaces, list);
                }
                root.surfaceOffset = original.surfaceOffset;
                root.surfaceCount  = original.surfaceCount;
                return builder.CreateBlobAssetReference<ChiselBrushRenderBuffer>(Allocator.Persistent);
            }
            finally
            {
                builder.Dispose();
            }
        }

        // A surface mapped with its group: cut with the group's sections (built from what the group draws together), or
        // given the group's tile as it is; one not in the group's first plane is unfolded into it for that (unfold), and
        // folded back, its own corners exactly where they were
        static void MapInGroup(BlobBuilder builder, ref ChiselSurfaceRenderBuffer from, ref ChiselSurfaceRenderBuffer to, in HotspotGroups.Group group, int groupIndex,
                               float4x4 unfold, bool unfolded,
                               NativeList<HotspotSectionLine> lines, NativeList<HotspotSectionHalfPlane> halfPlanes, NativeList<HotspotSectionCell> cells)
        {
            var indices  = ToList(ref from.indices);
            var collider = ToList(ref from.colliderVertices);
            var select   = ToList(ref from.selectVertices);
            var render   = ToList(ref from.renderVertices);
            var original = new UnsafeHashMap<float3, float3>(unfolded ? collider.Length : 1, Allocator.Temp);
            var budget   = group.input.MaxTriangles;
            var laid     = 0;
            var limited  = false;
            // with the interior's repeats; over the budget without them; still over, uncut on the group's bounds map
            for (int attempt = 0; attempt < 3; attempt++)
            {
                if (attempt > 0)
                {
                    Load(ref from, indices, collider, select, render);
                    original.Clear();
                }
                if (unfolded)
                {
                    for (int v = 0; v < collider.Length; v++)
                    {
                        var p = math.mul(unfold, new float4(collider[v], 1)).xyz;
                        original.TryAdd(p, collider[v]);
                        collider[v] = p;
                        var s = select[v]; s.position = p; select[v] = s;
                        var r = render[v]; r.position = p; render[v] = r;
                    }
                }
                if (group.sectionCount.z > 0 && attempt < 2)
                {
                    var sections = HotspotGroups.Sections(group, lines, halfPlanes, cells, Allocator.Temp);
                    laid = CutAndMap(sections, group.input, group.right, group.up, attempt == 0, indices, collider, select, render);
                    sections.Dispose();
                    if (indices.Length / 3 > budget)
                    {
                        limited = true;
                        continue;
                    }
                } else
                {
                    laid = 0;
                    MeshAlgorithms.ComputeUVs(render, group.uvMatrix);
                }
                break;
            }
            if (unfolded)
            {
                var back = math.inverse(unfold);
                for (int v = 0; v < collider.Length; v++)
                {
                    var p = original.TryGetValue(collider[v], out var exact) ? exact : math.mul(back, new float4(collider[v], 1)).xyz;
                    collider[v] = p;
                    var s = select[v]; s.position = p; select[v] = s;
                    var r = render[v]; r.position = p; render[v] = r;
                }
            }
            original.Dispose();
            MeshAlgorithms.ComputeTangents(indices, render);
            if (!Finite(collider, render))
            {
                to.CopyGeometry(builder, ref from);
                to.CopyHotspot(builder, ref from);
            } else
            {
                to.Store(builder, indices, collider, select, render);
                builder.Construct(ref to.physicsIndices,  ref from.indices);
                builder.Construct(ref to.physicsVertices, ref from.colliderVertices);
                to.CopySections(builder, ref from);
                to.sectionInput = group.input;
                to.hotspot      = group.info;
                to.hotspot.group = groupIndex + 1;
                to.hotspot.repeats = laid;
                if (limited)
                    to.hotspot.flags |= HotspotSurfaceFlags.BudgetLimited;
            }
            indices.Dispose(); collider.Dispose(); select.Dispose(); render.Dispose();
        }

        // What a surface draws cut where its sections meet (once; triangle by triangle when that would not cover it once),
        // and again where its interior's repeats meet (withRepeats), and given its uvs; how many repeats were laid
        static int CutAndMap(in HotspotSections sections, in HotspotSectionInput input, float3 right, float3 up, bool withRepeats,
                             NativeList<int> indices, NativeList<float3> collider, NativeList<SelectVertex> select, NativeList<RenderVertex> render)
        {
            var normal = math.normalizesafe(math.cross(up, right));
            if (!sections.CutOnce(right, up, normal, indices, collider, select, render))
                sections.Split(right, up, normal, indices, collider, select, render);
            var repeats = withRepeats ? InteriorRepeats(sections, input, right, up, normal, indices, collider, select, render) : HotspotSections.Create(Allocator.Temp);
            sections.AssignUVs(indices, collider, select, render, right, up, repeats);
            var laid = repeats.cells.IsCreated ? repeats.cells.Length : 0;
            repeats.Dispose();
            return laid;
        }

        // A surface's lists as the weld left them, again
        static void Load(ref ChiselSurfaceRenderBuffer from, NativeList<int> indices, NativeList<float3> collider, NativeList<SelectVertex> select, NativeList<RenderVertex> render)
        {
            indices.Clear(); collider.Clear(); select.Clear(); render.Clear();
            for (int i = 0; i < from.indices.Length; i++)          indices.Add(from.indices[i]);
            for (int i = 0; i < from.colliderVertices.Length; i++) collider.Add(from.colliderVertices[i]);
            for (int i = 0; i < from.selectVertices.Length; i++)   select.Add(from.selectVertices[i]);
            for (int i = 0; i < from.renderVertices.Length; i++)   render.Add(from.renderVertices[i]);
        }

        // A bordered tile's interior repeated at the tile's density (BuildInteriorRepeats): what is drawn cut again, triangle
        // by triangle, where the repeats meet (the border's strips too, so no edge has a point on one side only)
        static HotspotSections InteriorRepeats(in HotspotSections sections, in HotspotSectionInput input, float3 right, float3 up, float3 normal,
                                               NativeList<int> indices, NativeList<float3> collider, NativeList<SelectVertex> select, NativeList<RenderVertex> render)
        {
            var repeats = HotspotSections.Create(Allocator.Temp);
            if (sections.interiorCell < 0 || !repeats.BuildInteriorRepeats(input, sections.openSides, sections.interiorToEdges))
            {
                repeats.cells.Clear();
                return repeats;
            }
            if (repeats.cells.Length > 1 && !repeats.CutOnce(right, up, normal, indices, collider, select, render, asTriangles: true))
                repeats.Split(right, up, normal, indices, collider, select, render);
            return repeats;
        }

        static bool Finite(NativeList<float3> positions, NativeList<RenderVertex> render)
        {
            for (int i = 0; i < positions.Length; i++)
            {
                if (!math.all(math.isfinite(positions[i])))
                    return false;
            }
            for (int i = 0; i < render.Length; i++)
            {
                if (!math.all(math.isfinite(render[i].position)) || !math.all(math.isfinite(render[i].uv0)))
                    return false;
            }
            return true;
        }

        static NativeList<T> ToList<T>(ref BlobArray<T> array) where T : unmanaged
        {
            var list = new NativeList<T>(array.Length, Allocator.Temp);
            for (int i = 0; i < array.Length; i++)
                list.Add(array[i]);
            return list;
        }
    }
}
