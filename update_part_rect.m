function [ part_pos ] = update_part_rect( box, part_pos,npart )

width = box(3);
    height = box(4);
    if npart ==2
        if height>width
        part_pos{1}=[box(1)+width/2 box(2)+height/4];
        part_pos{2}=[box(1)+width/2 box(2)+height/2+height/4];

        else
        part_pos{1}=[box(1)+width/4 box(2)+height/2];
        part_pos{2}=[box(1)+width/2+width/4 box(2)+height/2];
        end
    elseif npart >2
        multiplier = sqrt(npart);
        if multiplier ~= fix(multiplier) 
            error('Invalid npart number!') ;
        end
        for i = 1:multiplier
            for j = 1:multiplier
               part_pos{(i-1)*j+j} =  [box(1)+(i-1)*width/multiplier+width/(2*multiplier) box(2)+(j-1)*height/multiplier+height/(2*multiplier)];
            end
        end
        
    else
        error('Invalid npart number!') ;
    end

  end

