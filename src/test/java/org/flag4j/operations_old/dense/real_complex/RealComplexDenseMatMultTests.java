package org.flag4j.operations_old.dense.real_complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations_old.dense.real_complex.RealComplexDenseMatrixMultiplicationOld.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class RealComplexDenseMatMultTests {
    double[][] aEntries;
    CNumber[][] expCEntries;

    MatrixOld A;
    CMatrixOld expC;

    @Test
    void matMultTestCase() {
        CNumber[][] bEntries;
        CMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrixOld(bEntries);
        expCEntries = new CNumber[][]{{new CNumber("1.8715844-91.6141568794i"), new CNumber("8768.731856002458-10.238294909999999i")},
                {new CNumber("-1553.4617-1447.705376035i"), new CNumber("37924.640881531595+8428.0382634i")},
                {new CNumber("205.65936999999997+123.444878510567i"), new CNumber("1419.6289704199999-1126.188735i")},
                {new CNumber("130.337844+66.8009098206i"), new CNumber("1978.9472913999998-846470.621682i")}};
        expC = new CMatrixOld(expCEntries);

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
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrixOld(bEntries);
        expCEntries = new CNumber[][]{{new CNumber("-10721.303415600001+8507.86475i"), new CNumber("6505.848844-4930.73205i"), new CNumber("-10.619125082+8.42448366i")},
                {new CNumber("-82273.7933-1.04985560794i"), new CNumber("48647.8159-92.7375568794i"), new CNumber("-81.47704028-0.00011494769430000002i")},
                {new CNumber("-0.018649000000000002-79472.71350000001i"), new CNumber("0.011027000000000002+46991.56050000001i"), new CNumber("-0.0000184684-78.7030866i")}};
        expC = new CMatrixOld(expCEntries);

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
        CNumber[][] bEntries;
        CMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new double[][]{{1.1234, 99.234, 0.000123},
                {-932.45, 551.35, -0.92342},
                {123.445, 0.00013, 0.0},
                {78.234, 12.234, -9923.23}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i")},
                {new CNumber("-0.0-0.9345341i")},
                {new CNumber("0.0")}};
        B = new CMatrixOld(bEntries);
        expCEntries = new CNumber[][]{{new CNumber("1.8715844-91.6141568794i")},
                {new CNumber("-1553.4617-1447.705376035i")},
                {new CNumber("205.65936999999997+123.444878510567i")},
                {new CNumber("130.337844+66.8009098206i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, blockedVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(A.entries, A.shape, B.entries, B.shape));
        assertArrayEquals(expC.entries, concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new double[][]{{1.1234},
                {-932.45}};
        A = new MatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrixOld(bEntries);
        expCEntries = new CNumber[][]{{new CNumber("-10721.303415600001+8507.86475i")},
                {new CNumber("-82273.7933-1.04985560794i")},
                {new CNumber("-0.018649000000000002-79472.71350000001i")}};
        expC = new CMatrixOld(expCEntries);

        assertArrayEquals(expC.entries, standardVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, blockedVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentStandardVector(B.entries, B.shape, A.entries, A.shape));
        assertArrayEquals(expC.entries, concurrentBlockedVector(B.entries, B.shape, A.entries, A.shape));
    }
}
