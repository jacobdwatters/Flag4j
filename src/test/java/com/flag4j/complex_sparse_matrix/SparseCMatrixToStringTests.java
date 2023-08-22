package com.flag4j.complex_sparse_matrix;

import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseCMatrixToStringTests {

    @Test
    void toStringTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        String exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.39424+0.26881i"), new CNumber("0.31325+0.34679i"), new CNumber("0.30908+0.33655i")};
        aRowIndices = new int[]{2, 3, 4};
        aColIndices = new int[]{2, 0, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);
        
        exp = "Full Shape: 5x3\n" +
                "Non-zero entries: [ 0.39424 + 0.26881i  0.31325 + 0.34679i  0.30908 + 0.33655i ]\n" +
                "Row Indices: [2, 3, 4]\n" +
                "Col Indices: [2, 0, 0]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.30062+0.9497i"), new CNumber("0.77614+0.28477i"), new CNumber("0.35101+0.73127i"), new CNumber("0.67145+0.68637i"), new CNumber("0.05538+0.33924i")};
        aRowIndices = new int[]{5, 6, 7, 8, 10};
        aColIndices = new int[]{12, 9, 12, 5, 12};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 11x23\n" +
                "Non-zero entries: [ 0.30062 + 0.9497i  0.77614 + 0.28477i  0.35101 + 0.73127i  0.67145 + 0.68637i  0.05538 + 0.33924i ]\n" +
                "Row Indices: [5, 6, 7, 8, 10]\n" +
                "Col Indices: [12, 9, 12, 5, 12]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 1000);
        aEntries = new CNumber[]{new CNumber("0.30698+0.99159i"), new CNumber("0.76603+0.52838i"), new CNumber("0.06296+0.54306i"), new CNumber("0.43915+0.00082i"), new CNumber("0.1874+0.22538i"), new CNumber("0.61855+0.69555i"), new CNumber("0.97349+0.45167i"), new CNumber("0.02954+0.5185i"), new CNumber("0.8994+0.8395i")};
        aRowIndices = new int[]{0, 1, 1, 1, 2, 2, 2, 4, 4};
        aColIndices = new int[]{736, 52, 123, 160, 180, 857, 868, 149, 899};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 5x1000\n" +
                "Non-zero entries: [ 0.30698 + 0.99159i  0.76603 + 0.52838i  0.06296 + 0.54306i  0.43915 + 8.2E-4i  0.1874 + 0.22538i  0.61855 + 0.69555i  0.97349 + 0.45167i  0.02954 + 0.5185i  0.8994 + 0.8395i ]\n" +
                "Row Indices: [0, 1, 1, 1, 2, 2, 2, 4, 4]\n" +
                "Col Indices: [736, 52, 123, 160, 180, 857, 868, 149, 899]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.24821+0.41705i"), new CNumber("0.22593+0.12134i"), new CNumber("0.37857+0.33477i"), new CNumber("0.56466+0.78808i")};
        aRowIndices = new int[]{0, 0, 2, 2};
        aColIndices = new int[]{1, 4, 2, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 3x5\n" +
                "Non-zero entries: [ 0.24821 + 0.41705i  0.22593 + 0.12134i  0.37857 + 0.33477i  0.56466 + 0.78808i ]\n" +
                "Row Indices: [0, 0, 2, 2]\n" +
                "Col Indices: [1, 4, 2, 3]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.95154+0.27456i"), new CNumber("0.84541+0.49608i"), new CNumber("0.93666+0.20043i"), new CNumber("0.65039+0.91006i")};
        aRowIndices = new int[]{0, 0, 2, 2};
        aColIndices = new int[]{2, 4, 0, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 3x5\n" +
                "Non-zero entries: [ 0.95154 + 0.27456i  0.84541 + 0.49608i  0.93666 + 0.20043i  0.65039 + 0.91006i ]\n" +
                "Row Indices: [0, 0, 2, 2]\n" +
                "Col Indices: [2, 4, 0, 3]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.44621+0.47313i"), new CNumber("0.93299+0.88628i"), new CNumber("0.6173+0.07362i"), new CNumber("0.13546+0.15639i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 0, 3, 4};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 3x5\n" +
                "Non-zero entries: [ 0.44621 + 0.47313i  0.93299 + 0.88628i  0.6173 + 0.07362i  0.13546 + 0.15639i ]\n" +
                "Row Indices: [0, 1, 1, 2]\n" +
                "Col Indices: [1, 0, 3, 4]";

        assertEquals(exp, a.toString());

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.38658+0.18229i"), new CNumber("0.17275+0.79699i"), new CNumber("0.08104+0.6007i"), new CNumber("0.31236+0.92982i")};
        aRowIndices = new int[]{0, 0, 1, 1};
        aColIndices = new int[]{0, 4, 2, 3};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        exp = "Full Shape: 3x5\n" +
                "Non-zero entries: [ 0.38658 + 0.18229i  0.17275 + 0.79699i  0.08104 + 0.6007i  0.31236 + 0.92982i ]\n" +
                "Row Indices: [0, 0, 1, 1]\n" +
                "Col Indices: [0, 4, 2, 3]";

        assertEquals(exp, a.toString());
    }
}
