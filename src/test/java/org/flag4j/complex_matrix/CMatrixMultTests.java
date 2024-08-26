package org.flag4j.complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixMultTests {
    CNumber[][] aEntries, expEntries;
    CMatrixOld A, exp;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        MatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new MatrixOld(bEntries);
        expEntries = new CNumber[][]{{new CNumber("163.51005868-15.462680014470001i"), new CNumber("5408.426908-109.8881922i")},
                {new CNumber("1.666+694.4522897100001i"), new CNumber("11.49931-65566.68726i")},
                {new CNumber("12690.663369999998"), new CNumber("87600.6175")},
                {new CNumber("-3.377522800415388-8.877571549119406i"), new CNumber("849.1747509679817+1368.4617155162825i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new MatrixOld(bEntries);

        MatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        CNumber[][] bEntries;
        CMatrixOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrixOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("215.01988001447+65.76525868i"), new CNumber("5323.5830080000005-776.3366921999999i")},
                {new CNumber("-692.78628971+1.0i"), new CNumber("7937.8893100000005-68516.24526000001i")},
                {new CNumber("12690.663369999998+7617.445i"), new CNumber("87600.6175-69493.95073499999i")},
                {new CNumber("16.203765617290802-0.235930146825595i"), new CNumber("877.8355007466813+998.8809657375828i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")}};
        B = new CMatrixOld(bEntries);

        CMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        CooMatrixOld B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooMatrixOld(bShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("-42.240941320000005+0.031119985530000005i"), new CNumber("63.018")},
                {new CNumber("0.0+694.4522897100001i"), new CNumber("-402.615-1085.31i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-8.611416161295983-14.11146491i"), new CNumber("-46.68")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooMatrixOld(bShape, bEntries, rowIndices, colIndices);

        CooMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        CNumber[] bEntries;
        int[] rowIndices, colIndices;
        CooCMatrixOld B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrixOld(bShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("-41.929586320000006+422.65111998553i"), new CNumber("63.018-10.8i")},
                {new CNumber("6947.985+694.4522897100001i"), new CNumber("-588.615-1016.31i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-149.79641616129598+72.04562781472501i"), new CNumber("-46.68+8.0i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new CooCMatrixOld(bShape, bEntries, rowIndices, colIndices);

        CooCMatrixOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void powTestCase() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO}};
        A = new CMatrixOld(aEntries);
        expEntries = new CNumber[][]{{new CNumber("120661.30796950031-33236207.159211755i"), new CNumber("-22728032.650843892-4235221.2280359i"), new CNumber("-3050628.73221+644215.430715i")},
                {new CNumber("-565975794.5347501+110147879.29020001i"), new CNumber("-11896701.7985705+378258502.7781383i"), new CNumber("17630791.944599997+47520631.43985i")},
                {new CNumber("429206180.168535-17498286.570418503i"), new CNumber("42331247.00392061-259089053.0570348i"), new CNumber("-6822160.1279205-32394488.588211752i")}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.pow(3));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO}};
        A = new CMatrixOld(aEntries);
        expEntries = new CNumber[][]{{new CNumber(1), CNumber.ZERO, CNumber.ZERO},
                {CNumber.ZERO, new CNumber(1), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO, new CNumber(1)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.pow(0));


        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO}};
        A = new CMatrixOld(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.pow(1));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)}};
        A = new CMatrixOld(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.pow(2));

        // ---------------------- Sub-case 5 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO}};

        assertThrows(IllegalArgumentException.class, ()->A.pow(-1));
    }
}
