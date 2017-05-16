#include "header.hpp"


// int main () {
//     string line;
//     //create an output stream to write to the file
//     //append the new lines to the end of the file
//     fstream myfileI ("test.txt", ios::app);
//     if (myfileI.is_open())
//     {
//         myfileI << "This file is for testing and learning";
//         myfileI << "\nI am adding a line.\n";
//         myfileI << "I am adding another line.\n";
//         myfileI.close();
//     }
//     else cout << "Unable to open file for writing";

//     //create an input stream to read the file
//     fstream myfileO ("input.txt");
//     //During the creation of ifstream, the file is opened. 
//     //So we do not have explicitly open the file. 
//     if (myfileO.is_open())
//     {
//         while ( getline (myfileO,line) )
//         {
//             cout << line << '\n';
//         }
//         myfileO.close();
//     }

//     else cout << "Unable to open file for reading";

//     return 0;
// }
// int main ()
 {
   std::string stringLength, stringWidth;
   float length = 0;
   float width = 0;
   float area = 0;

   std::cout << "Enter the length of the room: ";
   //get the length as a string
   std::getline (std::cin,stringLength);
   //convert to a float
   std::stringstream(stringLength) >> length;
   //get the width as a string
   std::cout << "Enter width: ";
   std::getline (std::cin,stringWidth);
   //convert to a float
   std::stringstream(stringWidth) >> width;
   area = length * width;
   std::cout << "\nThe area of the room is: " << area << std::endl;
   return 0;
 }
// int main ()
// {
//     cout << "hello";
//     return 0;
// }