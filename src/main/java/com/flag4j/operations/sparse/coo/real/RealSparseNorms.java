package com.flag4j.operations.sparse.coo.real;

import com.flag4j.CooMatrix;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;

import java.util.HashMap;

/**
 * This class contains low level implementations of norms for tensors, matrices and vector.
 */
public class RealSparseNorms {

    private RealSparseNorms() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the L<sub>2</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @return The L<sub>2</sub> of the {@code src} matrix.
     */
    public static double matrixNormL2(CooMatrix src) {
        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++) {
            colSums[columnMap.get(src.colIndices[i])] += src.entries[i]*src.entries[i];
        }

        // Compute the norm from the column sums.
        for(double colSum : colSums) {
            norm += Math.sqrt(colSum);
        }

        return norm;
    }


    /**
     * Computes the L<sub>p</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @param p Parameter for L<sub>p</sub> norm
     * @return The L<sub>p</sub> of the {@code src} matrix.
     */
    public static double matrixNormLp(CooMatrix src, double p) {
        ParameterChecks.assertGreaterEq(1, p);

        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++) {
            colSums[columnMap.get(src.colIndices[i])] += Math.pow(Math.abs(src.entries[i]), p);
        }

        // Compute the norm from the column sums.
        for(double colSum : colSums) {
            norm += Math.pow(colSum, 1.0/p);
        }

        return norm;
    }


    /**
     * Computes the L<sub>p, q</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @param p First parameter for L<sub>p, q</sub> norm
     * @return The L<sub>p, q</sub> of the {@code src} matrix.
     */
    public static double matrixNormLpq(CooMatrix src, double p, double q) {
        ParameterChecks.assertGreaterEq(1, p, q);

        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++) {
            colSums[columnMap.get(src.colIndices[i])] += Math.pow(Math.abs(src.entries[i]), p);
        }

        // Compute the norm from the column sums.
        for(double colSum : colSums) {
            norm += Math.pow(colSum, p/q);
        }

        return Math.pow(norm, 1.0/q);
    }
}
