package org.flag4j.complex_matrix;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixSubTests {

    Complex128[][] aEntries, expEntries;
    Shape sparseShape;
    int[] rowIndices, colIndices;
    CMatrix A, exp;


    @Test
    void realTestCase() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5},
                {345, 6.8883}};
        B = new Matrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub(B.entries[0]), new Complex128(4).sub(B.entries[1])},
                {new Complex128(67.1, 0.0003443993).sub(B.entries[2]), new Complex128(8.4554, -98.2).sub(B.entries[3])},
                {new Complex128(-723.234, 4).sub(B.entries[4]), new Complex128(-9.431).sub(B.entries[5])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, -0.4334},
                {0, 234.5}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23)},
                {new Complex128(67.1, 0.0003443993)},
                {new Complex128(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {123.235235, 234.5},
                {0, -0.43},
                {345, 45}};
        B = new Matrix(bEntries);

        Matrix finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void complexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.344, -0.3223), new Complex128(0)},
                {new Complex128(0, 213.57), new Complex128(-4941.3234)},
                {new Complex128(994.33134, Double.POSITIVE_INFINITY), new Complex128(445, 6)}};
        B = new CMatrix(bEntries);
        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub((Complex128) B.entries[0]), new Complex128(4).sub((Complex128) B.entries[1])},
                {new Complex128(67.1, 0.0003443993).sub((Complex128) B.entries[2]), new Complex128(8.4554, -98.2).sub((Complex128) B.entries[3])},
                {new Complex128(-723.234, 4).sub((Complex128) B.entries[4]), new Complex128(-9.431).sub((Complex128) B.entries[5])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.344, -0.3223), new Complex128(0)},
                {new Complex128(0, 213.57), new Complex128(-4941.3234)}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23)},
                {new Complex128(67.1, 0.0003443993)},
                {new Complex128(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[][]{
                {new Complex128(234.344, -0.3223), new Complex128(0)},
                {new Complex128(0, 213.57), new Complex128(-4941.3234)},
                {new Complex128(994.33134, Double.POSITIVE_INFINITY), new Complex128(445, 6)}};
        B = new CMatrix(bEntries);

        CMatrix finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub(B.entries[0]), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431).sub(B.entries[1])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+4);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23)},
                {new Complex128(67.1, 0.0003443993)},
                {new Complex128(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.45, -234.57};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+1, A.numCols);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooMatrix finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void complexSparseTestCase() {
        Complex128[] bEntries;
        CooCMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(234, -0.345), new Complex128(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub((Complex128) B.entries[0]), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431).sub((Complex128) B.entries[1])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(234, -0.345), new Complex128(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows+345, A.numCols+234);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23)},
                {new Complex128(67.1, 0.0003443993)},
                {new Complex128(-723.234, 4)}};
        A = new CMatrix(aEntries);
        bEntries = new Complex128[]{new Complex128(234, -0.345), new Complex128(0, 45.6)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{0, 1};
        sparseShape = new Shape(A.numRows, A.numCols+1000);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);

        CooCMatrix finalB1 = B;
        assertThrows(LinearAlgebraException.class,()->A.sub(finalB1));
    }


    @Test
    void doubleTestCase() {
        double b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        b = -0.234;
        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub(b), new Complex128(4).sub(b)},
                {new Complex128(67.1, 0.0003443993).sub(b), new Complex128(8.4554, -98.2).sub(b)},
                {new Complex128(-723.234, 4).sub(b), new Complex128(-9.431).sub(b)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(b));
    }


    @Test
    void complexNumberTestCase() {
        Complex128 b;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23), new Complex128(4)},
                {new Complex128(67.1, 0.0003443993), new Complex128(8.4554, -98.2)},
                {new Complex128(-723.234, 4), new Complex128(-9.431)}};
        A = new CMatrix(aEntries);
        b = new Complex128(-0.34534, 12.56);
        expEntries = new Complex128[][]{
                {new Complex128(234.56, -0.23).sub(b), new Complex128(4).sub(b)},
                {new Complex128(67.1, 0.0003443993).sub(b), new Complex128(8.4554, -98.2).sub(b)},
                {new Complex128(-723.234, 4).sub(b), new Complex128(-9.431).sub(b)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sub(b));
    }
}
