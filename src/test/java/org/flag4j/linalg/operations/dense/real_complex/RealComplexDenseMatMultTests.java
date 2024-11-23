package org.flag4j.linalg.operations.dense.real_complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseMatMult.*;

class RealComplexDenseMatMultTests {
    double[][] aEntries;
    Complex128[][] expCEntries;

    Matrix A;
    CMatrix expC;

    @Test
    void matMultTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("1.8715844-91.6141568794i"), new Complex128("8768.731856002458-10.238294909999999i")},
                {new Complex128("-1553.4617-1447.705376035i"), new Complex128("37924.640881531595+8428.0382634i")},
                {new Complex128("205.65936999999997+123.444878510567i"), new Complex128("1419.6289704199999-1126.188735i")},
                {new Complex128("130.337844+66.8009098206i"), new Complex128("1978.9472913999998-846470.621682i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standard(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, reordered(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, blocked(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, blockedReordered(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentStandard(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentReordered(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentBlocked(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("-10721.303415600001+8507.86475i"), new Complex128("6505.848844-4930.73205i"), new Complex128("-10.619125082+8.42448366i")},
                {new Complex128("-82273.7933-1.04985560794i"), new Complex128("48647.8159-92.7375568794i"), new Complex128("-81.47704028-0.00011494769430000002i")},
                {new Complex128("-0.018649000000000002-79472.71350000001i"), new Complex128("0.011027000000000002+46991.56050000001i"), new Complex128("-0.0000184684-78.7030866i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standard(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, reordered(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, blocked(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, blockedReordered(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandard(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentReordered(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentBlocked(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentBlockedReordered(B.entries, B.shape, A.entries, A.shape));
    }


    @Test
    void matVecMultRealComplexTestCase() {
        Complex128[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i")},
                {new Complex128("-0.0-0.9345341i")},
                {new Complex128("0.0")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("1.8715844-91.6141568794i")},
                {new Complex128("-1553.4617-1447.705376035i")},
                {new Complex128("205.65936999999997+123.444878510567i")},
                {new Complex128("130.337844+66.8009098206i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standardVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, blockedVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45}};
        A = new Matrix(aEntries);
        bEntries = new Complex128[][]{{new Complex128("1.666+1.0i"), new Complex128("11.5-9.123i")},
                {new Complex128("-0.0-0.9345341i"), new Complex128("88.234")},
                {new Complex128("0.0"), new Complex128("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expCEntries = new Complex128[][]{{new Complex128("-10721.303415600001+8507.86475i")},
                {new Complex128("-82273.7933-1.04985560794i")},
                {new Complex128("-0.018649000000000002-79472.71350000001i")}};
        expC = new CMatrix(expCEntries);

        assertArrayEquals(expC.entries, standardVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, blockedVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentBlockedVector(B.entries, B.shape, A.entries, A.shape));
    }
}
