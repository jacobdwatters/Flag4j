package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.SparseMatrix;
import com.flag4j.exceptions.LinearAlgebraException;
import com.flag4j.io.PrintOptions;
import com.flag4j.operations.sparse.real.RealSparseElementSearch;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.RandomTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>This class provides methods for computing the LU decomposition of a real sparse matrix.</p>
 * <p>The following decompositions are provided: {@code A=LU}, {@code PA=LU}, and {@code PAQ=LU}.</p>
 */
public class RealSparseLUDecomposition extends LUDecomposition<SparseMatrix> {

    private int[] sortedUniqueRows;
    private int[] sortedUniqueCols;


    /**
     * Constructs a LU decomposer to decompose the specified matrix using partial pivoting.
     */
    public RealSparseLUDecomposition() {
        super(Pivoting.PARTIAL.ordinal());
    }


    /**
     * Constructs a LU decomposer to decompose the specified matrix.
     *
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public RealSparseLUDecomposition(int pivoting) {
        super(pivoting);
    }


    /**
     * Initializes the {@code LU} matrix by copying the source matrix to decompose.
     *
     * @param src Source matrix to decompose.
     */
    @Override
    protected void initLU(SparseMatrix src) {
        LU = new SparseMatrix(src);
        sortedUniqueRows = ArrayUtils.uniqueSorted(LU.rowIndices);
        sortedUniqueCols = ArrayUtils.uniqueSorted(LU.colIndices);
    }


    /**
     * Computes the LU decomposition using no pivoting (i.e. rows and columns are not swapped).
     */
    @Override
    protected void noPivot() {

    }


    /**
     * Computes the LU decomposition using partial pivoting (i.e. row swapping).
     */
    @Override
    protected void partialPivot() {
        P = Matrix.I(LU.numRows);
        int colStop = Math.min(LU.numCols, LU.numRows);
        int maxIndex = 0;

        // Using Gaussian elimination with row pivoting.
        for(int j : sortedUniqueCols) {
            maxIndex = maxColIndex(j); // Find row index of max value (in absolute value) in column j so that the index >= j.

            // Make the appropriate swaps in LU and P (This is the partial pivoting step).
            if(j!=maxIndex && maxIndex>=0) {
                LU.swapRows(j, maxIndex);
                P.swapRows(j, maxIndex);
            }

            computeRows(j);
        }
    }


    /**
     * Computes the LU decomposition using full/rook pivoting (i.e. row and column swapping).
     */
    @Override
    protected void fullPivot() {

    }


    /**
     * Gets the unit lower triangular matrix of the decomposition.
     *
     * @return The unit lower triangular matrix of the decomposition.
     */
    @Override
    public SparseMatrix getL() {
        int initCap = LU.entries.length/2;

        Shape shape = new Shape(LU.numRows, Math.min(LU.numRows, LU.numCols));
        List<Double> lEntries = new ArrayList<>(initCap);
        List<Integer> lRowIndices = new ArrayList<>(initCap);
        List<Integer> lColIndices = new ArrayList<>(initCap);

        // Extract lower portion.
        for(int i=0; i<LU.entries.length; i++) {
            if(LU.rowIndices[i] > LU.colIndices[i]) {
                lEntries.add(LU.entries[i]);
                lRowIndices.add(LU.rowIndices[i]);
                lColIndices.add(LU.colIndices[i]);
            }
        }

        // Set diagonal to ones.
        for(int i=0; i<shape.get(0); i++) {
            lEntries.add(1.0);
            lRowIndices.add(i);
            lColIndices.add(i);
        }

        SparseMatrix L = new SparseMatrix(shape, lEntries, lRowIndices, lColIndices);
        L.sparseSort();

        return L;
    }


    /**
     * Gets the upper triangular matrix of the decomposition.
     *
     * @return The upper triangular matrix of the decomposition.
     */
    @Override
    public SparseMatrix getU() {
        int initCap = LU.entries.length/2;

        Shape shape = new Shape(Math.min(LU.numRows, LU.numCols), LU.numCols);
        List<Double> uEntries = new ArrayList<>(initCap);
        List<Integer> uRowIndices = new ArrayList<>(initCap);
        List<Integer> uColIndices = new ArrayList<>(initCap);

        // Extract lower portion.
        for(int i=0; i<LU.entries.length; i++) {
            if(LU.rowIndices[i] <= LU.colIndices[i]) {
                uEntries.add(LU.entries[i]);
                uRowIndices.add(LU.rowIndices[i]);
                uColIndices.add(LU.colIndices[i]);
            }
        }

        SparseMatrix U = new SparseMatrix(shape, uEntries, uRowIndices, uColIndices);
        U.sparseSort();

        return U;
    }


    /**
     * Computes the max absolute value in a column so that the row is >= j.
     * @param j column index.
     * @return The index of the maximum absolute value in the specified column such that the row is >= j.
     */
    private int maxColIndex(int j) {
        int maxIndex = -1;
        double currentMax = -1;
        double value;

        int rowIdx = ArrayUtils.indexOf(LU.rowIndices, j);

        if(rowIdx!=-1) {
            for(int i=rowIdx; i<LU.entries.length; i++) {
                if(LU.colIndices[i] == j) {
                    value = Math.abs(LU.entries[i]);

                    if(value > currentMax) {
                        currentMax = value;
                        maxIndex = LU.rowIndices[i];
                    }
                }
            }
        }

        return maxIndex;
    }


    /**
     * Helper method which computes rows in the gaussian elimination algorithm.
     * @param j Column for which to compute values to the right of.
     */
    private void computeRows(int j) {
        double m;
        int pivotRow = j*LU.numCols;
        int iRow;

        int rowStart = ArrayUtils.indexOf(sortedUniqueRows, j+1);

        if(rowStart != -1) {
            for(int i=rowStart; i<LU.entries.length; i++) {
                double q = LU.get(j, j);
                m = LU.get(LU.rowIndices[i], j);
                m = q==0 ? m : m/q;

                int[] startEnd = RealSparseElementSearch.matrixFindRowStartEnd(LU, LU.rowIndices[i]);

                for(int k=startEnd[0]; k<startEnd[1]; k++) {
                    LU.entries[k] -= m*LU.get(j, k);
                }

                if(m != 0) {
                    LU = LU.set(m, LU.rowIndices[i], j);
                }
            }
        }

//        for(int i=j+1; i<LU.numRows; i++) {
//            iRow = i*LU.numCols;
//            m = LU.entries[iRow + j]; // m = LU[i, j]
//            m = LU.entries[pivotRow + j] == 0 ? m : m/LU.entries[pivotRow + j]; // m = LU[j, j]==0 ? m : m / LU[j, j]
//
//            // Compute and set U values.
//            for(int k=j; k<LU.numCols; k++) {
//                LU.entries[iRow + k] -= m*LU.entries[pivotRow + k];
//            }
//
//            // Compute and set L value.
//            LU.entries[iRow + j] = m;
//        }
    }


    public static void main(String[] args) {
        PrintOptions.setMaxRowsCols(20);
        RandomTensor rtg = new RandomTensor(2309);
        SparseMatrix A = rtg.randomSparseMatrix(10, 10, -100, 100,0.9);

        System.out.println("A:\n" + A.toDense() + "\n\n");

        RealSparseLUDecomposition lu = new RealSparseLUDecomposition();
        lu.decompose(A);

        SparseMatrix L = lu.getL();
        SparseMatrix U = lu.getU();
        Matrix P = lu.getP();

        System.out.println("P:\n" + P + "\n\n");
        System.out.println("L:\n" + L.toDense() + "\n\n");
        System.out.println("U:\n" + U.toDense() + "\n\n");

        System.out.println("PA:\n" + P.mult(A) + "\n\n");
        System.out.println("LU:\n" + L.mult(U) + "\n\n");


        RealLUDecomposition denselu = new RealLUDecomposition();
        denselu.decompose(A.toDense());

        Matrix denseL = denselu.getL();
        Matrix denseU = denselu.getU();
        Matrix denseP = denselu.getP();

        System.out.println("DENSE:\n");
        System.out.println("P:\n" + denseP + "\n\n");
        System.out.println("L:\n" + denseL + "\n\n");
        System.out.println("U:\n" + denseU + "\n\n");

        System.out.println("PA:\n" + denseP.mult(A.toDense()) + "\n\n");
        System.out.println("LU:\n" + denseL.mult(denseU) + "\n\n");
    }
}
