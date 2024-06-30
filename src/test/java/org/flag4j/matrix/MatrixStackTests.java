package org.flag4j.matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Matrix;
import org.flag4j.dense.Vector;
import org.flag4j.sparse.CooCMatrix;
import org.flag4j.sparse.CooCVector;
import org.flag4j.sparse.CooMatrix;
import org.flag4j.sparse.CooVector;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MatrixStackTests {

    double[][] expEntries;
    CNumber[][] expCEntries;

    double[][] entries;
    double[][] entries2;
    CNumber[][] CEntries;

    double[] sparseEntries;
    CNumber[] sparseCEntries;

    int[] indices;
    int[] rowIndices;
    int[] colIndices;

    Shape sparseShape;

    Matrix realDense;
    Matrix realDense2;
    CooMatrix realSparse;
    CMatrix complexDense;
    CooCMatrix complexSparse;
    Vector realDenseVector;
    CooVector realCooVector;
    CVector complexDenseVector;
    CooCVector complexSparseVector;

    Matrix exp;
    CMatrix expC;

    @Test
    void realDenseTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}, {99.2134, -0.23}};

        expEntries = new double[][]{{1, 2, 3, 4, 5}, {4, 14, -5.12, 5, 4}, {0, 9.34, -0.13, 99.2134, -0.23}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 0));


        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5, -0.53}, {5, 4, 1.3}};

        expEntries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}, {4, 5, -0.53}, {5, 4, 1.3}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 7));
    }


    @Test
    void complexDenseTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, -0.98824)}, {new CNumber(575, -1.13)},
                {new CNumber(0, 1.33)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(1, -0.98824)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12), new CNumber(575, -1.13)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13), new CNumber(0, 1.33)}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertEquals(expC, realDense.stack(complexDense, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13)},
                {new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}
        };
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertEquals(expC, realDense.stack(complexDense, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 7));
    }


    @Test
    void realSparseTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1, 2, 3};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(3, 2);

        expEntries = new double[][]{
                {1, 2, 3, 1, 2},
                {4, 14, -5.12, 0, 0},
                {0, 9.34, -0.13, 0, 3}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realSparse = new CooMatrix(sparseShape, sparseEntries, rowIndices, colIndices);

        assertEquals(exp, realDense.stack(realSparse, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1, 2, 3};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(5, 2);

        realDense = new Matrix(entries);
        realSparse = new CooMatrix(sparseShape, sparseEntries, rowIndices, colIndices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realSparse, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1, 2, 3};
        rowIndices = new int[]{0, 2, 4};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(6, 3);

        expEntries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13},
                {1, 0, 0},
                {0, 0, 0},
                {0, 2, 0},
                {0, 0, 0},
                {0, 3, 0},
                {0, 0, 0}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realSparse = new CooMatrix(sparseShape, sparseEntries, rowIndices, colIndices);

        assertEquals(exp, realDense.stack(realSparse, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1, 2, 3};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(5, 144);

        realDense = new Matrix(entries);
        realSparse = new CooMatrix(sparseShape, sparseEntries, rowIndices, colIndices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realSparse, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realSparse, 7));
    }


    @Test
    void complexSparseTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2), new CNumber(1, 8.34), new CNumber(-93)};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(3, 2);

        expCEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(81, -9.2), new CNumber(1, 8.34)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(9.34), new CNumber(-0.13), new CNumber(), new CNumber(-93)}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexSparse = new CooCMatrix(sparseShape, sparseCEntries, rowIndices, colIndices);

        assertEquals(expC, realDense.stack(complexSparse, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2), new CNumber(1, 8.34), new CNumber(-93)};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(5, 2);

        realDense = new Matrix(entries);
        complexSparse = new CooCMatrix(sparseShape, sparseCEntries, rowIndices, colIndices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparse, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2), new CNumber(1, 8.34), new CNumber(-93)};
        rowIndices = new int[]{0, 2, 4};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(6, 3);

        expCEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13)},
                {new CNumber(81, -9.2), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1, 8.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(-93), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0)}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexSparse = new CooCMatrix(sparseShape, sparseCEntries, rowIndices, colIndices);

        assertEquals(expC, realDense.stack(complexSparse, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2), new CNumber(1, 8.34), new CNumber(-93)};
        rowIndices = new int[]{0, 0, 2};
        colIndices = new int[]{0, 1, 1};
        sparseShape = new Shape(5, 144);

        realDense = new Matrix(entries);
        complexSparse = new CooCMatrix(sparseShape, sparseCEntries, rowIndices, colIndices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparse, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparse, 7));
    }


    @Test
    void realDenseVectorTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4}, {5}, {99.2134}};

        expEntries = new double[][]{{1, 2, 3, 4}, {4, 14, -5.12, 5}, {0, 9.34, -0.13, 99.2134}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDenseVector = new Vector(ArrayUtils.flatten(entries2));

        assertEquals(exp, realDense.stack(realDenseVector, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDenseVector = new Vector(ArrayUtils.flatten(entries2));

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDenseVector, 0));


        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5, -0.53}};

        expEntries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}, {4, 5, -0.53}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDenseVector = new Vector(ArrayUtils.flatten(entries2));

        assertEquals(exp, realDense.stack(realDenseVector, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDenseVector = new Vector(ArrayUtils.flatten(entries2));

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDenseVector, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDenseVector, 7));
    }


    @Test
    void complexDenseVectorTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, -0.98824)}, {new CNumber(575, -1.13)},
                {new CNumber(0, 1.33)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(1, -0.98824)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12), new CNumber(575, -1.13)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13), new CNumber(0, 1.33)}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDenseVector = new CVector(ArrayUtils.flatten(CEntries));

        assertEquals(expC, realDense.stack(complexDenseVector, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDenseVector = new CVector(ArrayUtils.flatten(CEntries));

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDenseVector, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13)},
                {new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}
        };
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDenseVector = new CVector(ArrayUtils.flatten(CEntries));

        assertEquals(expC, realDense.stack(complexDenseVector, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDenseVector = new CVector(ArrayUtils.flatten(CEntries));

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDenseVector, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 7));
    }


    @Test
    void realSparseVectorTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1.123};
        indices = new int[]{1};

        expEntries = new double[][]{
                {1, 2, 3, 0},
                {4, 14, -5.12, 1.123},
                {0, 9.34, -0.13, 0}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realCooVector = new CooVector(3, sparseEntries, indices);

        assertEquals(exp, realDense.stack(realCooVector, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{1.123};
        indices = new int[]{1};

        realDense = new Matrix(entries);
        realCooVector = new CooVector(15, sparseEntries, indices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realCooVector, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{4.3};
        indices = new int[]{2};

        expEntries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13},
                {0, 0, 4.3}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realCooVector = new CooVector(3, sparseEntries, indices);

        assertEquals(exp, realDense.stack(realCooVector, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseEntries = new double[]{4.3};
        indices = new int[]{2};

        realDense = new Matrix(entries);
        realCooVector = new CooVector(22, sparseEntries, indices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realCooVector, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realCooVector, 7));
    }


    @Test
    void complexSparseVectorTestCase() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2)};
        indices = new int[]{0};

        expCEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(81, -9.2)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12), new CNumber()},
                {new CNumber(), new CNumber(9.34), new CNumber(-0.13), new CNumber()}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexSparseVector = new CooCVector(3, sparseCEntries, indices);

        assertEquals(expC, realDense.stack(complexSparseVector, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2)};
        indices = new int[]{0};

        realDense = new Matrix(entries);
        complexSparseVector = new CooCVector(3011, sparseCEntries, indices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparseVector, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2)};
        indices = new int[]{1};

        expCEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13)},
                {new CNumber(), new CNumber(81, -9.2), new CNumber()}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexSparseVector = new CooCVector(3, sparseCEntries, indices);

        assertEquals(expC, realDense.stack(complexSparseVector, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{
                {1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        sparseCEntries = new CNumber[]{new CNumber(81, -9.2)};
        indices = new int[]{1};

        realDense = new Matrix(entries);
        complexSparseVector = new CooCVector(2, sparseCEntries, indices);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparseVector, 1));

        // -----------------------  Sub-case 5 -----------------------
        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexSparseVector, 7));
    }
}
