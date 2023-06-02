package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixMultTests {
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void matMultTests() {
        double[][] bEntries;
        Matrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber("163.51005868-15.462680014470001i"), new CNumber("5408.426908-109.8881922i")},
                {new CNumber("1.666+694.4522897100001i"), new CNumber("11.49931-65566.68726i")},
                {new CNumber("12690.663369999998"), new CNumber("87600.6175")},
                {new CNumber("-3.377522800415388-8.877571549119406i"), new CNumber("849.1747509679817+1368.4617155162825i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTests() {
        CNumber[][] bEntries;
        CMatrix B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")},
                {new CNumber("0.0"), new CNumber("0.00002+85.23i")}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("215.01988001447+65.76525868i"), new CNumber("5323.5830080000005-776.3366921999999i")},
                {new CNumber("-692.78628971+1.0i"), new CNumber("7937.8893100000005-68516.24526000001i")},
                {new CNumber("12690.663369999998+7617.445i"), new CNumber("87600.6175-69493.95073499999i")},
                {new CNumber("16.203765617290802-0.235930146825595i"), new CNumber("877.8355007466813+998.8809657375828i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTests() {
        double[] bEntries;
        int[] rowIndices, colIndices;
        SparseMatrix B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("-42.240941320000005+0.031119985530000005i"), new CNumber("63.018")},
                {new CNumber("0.0+694.4522897100001i"), new CNumber("-402.615-1085.31i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-8.611416161295983-14.11146491i"), new CNumber("-46.68")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-0.9345341, 11.67};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new SparseMatrix(bShape, bEntries, rowIndices, colIndices);

        SparseMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTests() {
        CNumber[] bEntries;
        int[] rowIndices, colIndices;
        SparseCMatrix B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("-41.929586320000006+422.65111998553i"), new CNumber("63.018-10.8i")},
                {new CNumber("6947.985+694.4522897100001i"), new CNumber("-588.615-1016.31i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("-149.79641616129598+72.04562781472501i"), new CNumber("-46.68+8.0i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1, 2};
        colIndices = new int[]{0, 1};
        bShape = new Shape(31, 2);
        B = new SparseCMatrix(bShape, bEntries, rowIndices, colIndices);

        SparseCMatrix finalB = B;
        assertThrows(IllegalArgumentException.class, ()->A.mult(finalB));
    }


    @Test
    void powTests() {
        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{{new CNumber("120661.30796950031-33236207.159211755i"), new CNumber("-22728032.650843892-4235221.2280359i"), new CNumber("-3050628.73221+644215.430715i")},
                {new CNumber("-565975794.5347501+110147879.29020001i"), new CNumber("-11896701.7985705+378258502.7781383i"), new CNumber("17630791.944599997+47520631.43985i")},
                {new CNumber("429206180.168535-17498286.570418503i"), new CNumber("42331247.00392061-259089053.0570348i"), new CNumber("-6822160.1279205-32394488.588211752i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.pow(3));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{{new CNumber(1), new CNumber(), new CNumber()},
                {new CNumber(), new CNumber(1), new CNumber()},
                {new CNumber(), new CNumber(), new CNumber(1)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.pow(0));


        // ---------------------- Sub-case 3 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.pow(1));

        // ---------------------- Sub-case 4 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)}};
        A = new CMatrix(aEntries);

        assertThrows(IllegalArgumentException.class, ()->A.pow(2));

        // ---------------------- Sub-case 5 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()}};

        assertThrows(IllegalArgumentException.class, ()->A.pow(-1));
    }
}
