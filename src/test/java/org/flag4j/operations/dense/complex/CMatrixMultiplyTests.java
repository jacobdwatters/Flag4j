package org.flag4j.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CMatrixMultiplyTests {
    Complex128[][] entriesA, entriesB;
    CMatrix A, B;
    Complex128[] exp, act;


    @Test
    void squareTestCase() {
        entriesA = new Complex128[][]{
                {new Complex128(1.234, -0.0924), new Complex128(1.345), new Complex128(0, 8.55)},
                {new Complex128(-234.5), new Complex128(9.24, 6.772), new Complex128(-93.1567, -22.7)},
                {Complex128.ZERO, new Complex128(0, 0.2224), new Complex128(-2.56, 34)}};
        entriesB = new Complex128[][]{
                {new Complex128(0.924, 515.2), new Complex128(5, 1), new Complex128(0, 0.3)},
                {new Complex128(7.55, -0.009824), new Complex128(0.2245), new Complex128(90.24)},
                {new Complex128(4.5), new Complex128(9.14, 511.45), new Complex128(8, -7.14)}};
        exp = new Complex128[] {
                new Complex128("58.899446000000005+674.1332091200001i"), new Complex128("-4366.3331475+78.91900000000001i"), new Complex128("182.44752+68.7702i"),
                new Complex128("-566.054621872-120865.51217376i"), new Complex128("9588.037142-48085.451901i"), new Complex128("-73.51400000000001+1024.2941179999998i"),
                new Complex128("-11.5178151424+154.67912i"), new Complex128("-17412.698399999997-998.5020711999999i"), new Complex128("222.28+310.34777599999995i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = ComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void rectangleTestCase() {
        entriesA = new Complex128[][]{{new Complex128("1.234-0.0924i"), new Complex128("1.345")},
                {new Complex128("-234.5"), new Complex128("9.24+6.772i")},
                {new Complex128("0.0"), new Complex128("0.0+0.2224i")}};
        entriesB = new Complex128[][]{{new Complex128("0.924+515.2i"), new Complex128("5.0+1.0i"), new Complex128("0.0+0.3i")},
                {new Complex128("7.55-0.009824i"), new Complex128("0.2245"), new Complex128("90.24")}};
        exp = new Complex128[]{
                new Complex128("58.899446000000005+635.65820912i"), new Complex128("6.564352499999999+0.772i"),
                new Complex128("121.40051999999999+0.3702i"), new Complex128("-146.84947187199998 - 120763.36217376i"),
                new Complex128("-1170.42562-232.979686i"), new Complex128("833.8176+540.75528i"),
                new Complex128("0.0021848576+1.67912i"), new Complex128("0.0+0.049928799999999995i"), new Complex128("0.0+20.069376i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 5 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 6 ------------
        act = ComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 7 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 8 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }


    @Test
    void columnVectorTestCase() {
        entriesA = new Complex128[][]{
                {new Complex128(1.234, -0.0924), new Complex128(1.345), new Complex128(0, 8.55)},
                {new Complex128(-234.5), new Complex128(9.24, 6.772), new Complex128(-93.1567, -22.7)},
                {Complex128.ZERO, new Complex128(0, 0.2224), new Complex128(-2.56, 34)}};
        entriesB = new Complex128[][]{
                {new Complex128(0.924, 515.2)},
                {new Complex128(7.55, -0.009824)},
                {new Complex128(4.5)}};
        exp = new Complex128[]{new Complex128("58.899446000000005+674.1332091200001i"),
                new Complex128("-566.054621872-120865.51217376i"), new Complex128("-11.5178151424+154.67912i")};

        A = new CMatrix(entriesA);
        B = new CMatrix(entriesB);

        // ------------ Sub-case 1 ------------
        act = ComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 2 ------------
        act = ComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 3 ------------
        act = ComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);

        // ------------ Sub-case 4 ------------
        act = ComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
        assertArrayEquals(exp, act);
    }
}
