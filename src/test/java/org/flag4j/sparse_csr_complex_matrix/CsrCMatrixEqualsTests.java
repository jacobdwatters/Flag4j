package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class CsrCMatrixEqualsTests {
    CsrCMatrix A;
    CsrCMatrix B;

    CNumber[][] aEntries;
    CNumber[][] bEntries;

    CNumber[] aNnz;
    CNumber[] bNnz;
    int[][] aIndices;
    int[][] bIndices;
    Shape aShape;
    Shape bShape;

    @Test
    void realCsrEqualsTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[150][235];
        ArrayUtils.fill(aEntries, CNumber.ZERO);
        bEntries = new CNumber[150][235];
        ArrayUtils.fill(bEntries, CNumber.ZERO);
        aEntries[0][1] = new CNumber(134.4, -0.0234);
        aEntries[15][234] = new CNumber(-0.00024, 1.45);
        aEntries[49][1] = new CNumber(234.25000024, 234.5);
        aEntries[59][0] = new CNumber(0, 23.342);
        aEntries[59][2] = new CNumber(-0.0000005, 235.03);
        aEntries[149][1] = new CNumber(15);

        bEntries[0][1] = new CNumber(134.4, -0.0234);
        bEntries[15][234] = new CNumber(-0.00024, 1.45);
        bEntries[49][1] = new CNumber(234.25000024, 234.5);
        bEntries[59][0] = new CNumber(0, 23.342);
        bEntries[59][2] = new CNumber(-0.0000005, 235.03);
        bEntries[149][1] = new CNumber(15);

        A = new CMatrixOld(aEntries).toCsr();
        B = new CMatrixOld(bEntries).toCsr();

        assertEquals(A, B);

        // ---------------------- Sub-case 2 ----------------------
        aNnz = new CNumber[]{new CNumber(1.34507518, -352), new CNumber(0.235, 2), new CNumber(0, 72735),
                new CNumber(94.1, -1), new CNumber(1, 3), new CNumber(-1, 0.002345), new CNumber(15)};
        bNnz = new CNumber[]{new CNumber(1.34507518, -352), new CNumber(0.235, 2), new CNumber(0, 72735),
                new CNumber(94.1, -1), new CNumber(1, 3), new CNumber(-1, 0.002345), new CNumber(15)};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertEquals(A, B);

        // ---------------------- Sub-case 3 ----------------------
        aNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, new CNumber(345.1, 2.5), new CNumber(9.4, -1),
                new CNumber(235.1, 94.2), new CNumber(3.12, 4), CNumber.ZERO,
                new CNumber(0, 1), new CNumber(2,9733)};
        bNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(345.1, 2.5),
                new CNumber(9.4, -1),
                CNumber.ZERO, new CNumber(235.1, 94.2), new CNumber(3.12, 4),
                new CNumber(0, 1), new CNumber(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertEquals(A, B);

        // ---------------------- Sub-case 4 ----------------------
        aNnz = new CNumber[]{new CNumber(234.5, -0.2), new CNumber(4.23, 9), new CNumber(345.1, 2.5), new CNumber(9.4, -1),
                new CNumber(235.1, 94.2), new CNumber(3.12, 4), CNumber.ZERO,
                new CNumber(0, 1), new CNumber(2,9733)};
        bNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(345.1, 2.5),
                new CNumber(9.4, -1),
                CNumber.ZERO, new CNumber(235.1, 94.2), new CNumber(3.12, 4),
                new CNumber(0, 1), new CNumber(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- Sub-case 5 ----------------------
        aNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, new CNumber(345.1, 2.5), new CNumber(9.4, -1),
                new CNumber(235.1, 94.2), new CNumber(3.12, 4), CNumber.ZERO,
                new CNumber(0, 1), new CNumber(2,9733)};
        bNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(345.1, 2.5),
                new CNumber(9.4, -1),
                CNumber.ZERO, new CNumber(235.1, 94.2), new CNumber(3.12, 4),
                new CNumber(0, 1), new CNumber(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 3, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 45, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- Sub-case 6 ----------------------
        aNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, new CNumber(345.1, 2.5), new CNumber(9.4, -1),
                new CNumber(235.1, 94.2), new CNumber(3.12, 4), CNumber.ZERO,
                new CNumber(0, 1), new CNumber(2,9733)};
        bNnz = new CNumber[]{new CNumber(234.5, -0.2), CNumber.ZERO, CNumber.ZERO, CNumber.ZERO, new CNumber(345.1, 2.5),
                new CNumber(9.4, -1),
                CNumber.ZERO, new CNumber(235.1, 94.2), new CNumber(3.12, 4),
                new CNumber(0, 1), new CNumber(2,9733)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 5, 12, 14, 67, 67},
                {0, 2, 5, 14, 5002, 142, 55, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0, 1, 5, 5, 12, 67, 67},
                {0, 1, 2, 3, 5, 14, 22, 5008, 142, 15, 60001}};
        aShape = new Shape(900, 450000);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);

        // ---------------------- Sub-case 7 ----------------------
        aNnz = new CNumber[]{new CNumber(1.34507518, -352), new CNumber(0.235, 2), new CNumber(0, 72735),
                new CNumber(94.1, -1), new CNumber(1, 3), new CNumber(-1, 0.002345), new CNumber(15)};
        bNnz = new CNumber[]{new CNumber(1.34507518, -352), new CNumber(0.235, 2), new CNumber(0, 72735),
                new CNumber(94.1, -1), new CNumber(1, 3), new CNumber(-1, 0.002345), new CNumber(15)};
        aIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        bIndices = new int[][]{
                {0, 0, 1, 5, 12, 67, 67},
                {4, 5, 14, 5002, 142, 15, 60001}};
        aShape = new Shape(900, 60002);
        bShape = new Shape(900, 450000);
        A = new CooCMatrix(aShape, aNnz, aIndices[0], aIndices[1]).toCsr();
        B = new CooCMatrix(bShape, bNnz, bIndices[0], bIndices[1]).toCsr();

        assertNotEquals(A, B);
    }
}
