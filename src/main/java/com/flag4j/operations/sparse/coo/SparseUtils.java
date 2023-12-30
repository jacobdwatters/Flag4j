package com.flag4j.operations.sparse.coo;

import com.flag4j.util.ErrorMessages;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Contains common utility functions for working with sparse matrices.
 */
public final class SparseUtils {

    public SparseUtils() {
        // Utility class cannot be instanced.
        throw new IllegalArgumentException(ErrorMessages.getUtilityClassErrMsg());
    }



    /**
     * Creates a HashMap where the keys are row indices and the value is a list of all indices in src with that row
     * index.
     * @param nnz Number of non-zero entries in the sparse matrix.
     * @param rowIndices Row indices of sparse matrix.
     * @return A HashMap where the keys are row indices and the value is a list of all indices in {@code src} with that row
     * index.
     */
    public static Map<Integer, List<Integer>> createMap(int nnz, int[] rowIndices) {
        Map<Integer, List<Integer>> map = new HashMap<>();

        for(int j=0; j<nnz; j++) {
            int r2 = rowIndices[j]; // = k
            map.computeIfAbsent(r2, x -> new ArrayList<>()).add(j);
        }

        return map;
    }
}
